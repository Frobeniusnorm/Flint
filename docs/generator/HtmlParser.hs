module HtmlParser where
    import Data.Char (isDigit, isAlpha)
    import Data.List
    highKeyword = "#F030FF"
    highType = "#FFF030"
    highLiteral = "#30F0FF"

    replaceIllegal ('<':s) = "&lt;" ++ replaceIllegal s
    replaceIllegal ('>':s) = "&gt;" ++ replaceIllegal s
    replaceIllegal (x:s) = x : replaceIllegal s
    replaceIllegal [] = []

    highlightLiterals str =
        helper str True
        where
            helper (x:t) p
                | isAlpha x = x : helper t False
                | isDigit x && p =
                    "<span style=\"color: " ++ highLiteral ++ "\">" ++ x : "</span>" ++ helper t True
                | isDigit x = x : helper t False
                | otherwise = x : helper t True
            helper [] _ = []

    foldHtml :: FilePath -> IO ()
    foldHtml path = do
        content <- readFile path
        writeFile (newFilename path) (transformCode content)
        where
            transformCode str = concatMap (\x -> do
                    let code = map fst x
                    if snd (head x) then
                        highlightCode $ highlightLiterals $ replaceIllegal code
                    else code) 
                (groupBy (\a b -> snd a == snd b) (markCode str 0 False))
            markCode "" n _ = []
            markCode ('@':'c':'o':'d':'e':'{' :t) 0 _ = markCode t 1 True
            markCode ('{':t) n iscode = ('{', iscode) : markCode t (n + 1) iscode
            markCode ('}':t) 1 True = markCode t 0 False
            markCode ('}':t) n iscode = ('}',iscode) : markCode t (n - 1) iscode
            markCode (x : t) n iscode = (x, iscode) : markCode t n iscode

            highlightCode [] = [] 
            highlightCode ('c':'l':'a':'s':'s':t) =
                    "<span style=\"color: " ++ highKeyword ++ "\">class</span>" ++ highlightCode t
            highlightCode ('i':'n':'t':t) = 
                    "<span style=\"color: " ++ highType ++ "\">int</span>" ++ highlightCode t
            highlightCode ('T':'e':'n':'s':'o':'r':t) =
                    "<span style=\"color: " ++ highType ++ "\">Tensor</span>" ++ highlightCode t
            highlightCode ('/':'/':t) =
                    "<span style=\"color: #D0D0D0\">" ++ takeWhile (/= '\n') t ++ "</span>" ++ highlightCode (dropWhile (/= '\n') t)
            highlightCode ('/':'*':t) = do
                    -- char and following char zipped together to detect /*
                    let interleaved = zip t (drop 1 t)
                    let comment = map fst (takeWhile (\x -> fst x /= '*' || snd x /= '/') interleaved)
                    let rest = drop 2 (map fst (dropWhile (\x -> fst x /= '*' || snd x /= '/') interleaved))
                    "<span style=\"color: #D0D0D0\">/*" ++ comment ++
                        "*/</span>" ++ highlightCode rest
            -- here the replacement for < > happens
            highlightCode (x:t)= x : highlightCode t

            newFilename x = take (length x - 5) x  ++ "html"
