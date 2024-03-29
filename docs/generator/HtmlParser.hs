{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
module HtmlParser where
    import Data.Char (isDigit, isAlpha)
    import Data.List

    import CPPParser (compileCppToHtml, compileTOCForCPP, parseCpp, selectFcts)
    highKeyword = "#F030FF"
    highType = "#FFF030"
    highLiteral = "#30F0FF"

    replaceIllegal ('<':s) = "&lt;" ++ replaceIllegal s
    replaceIllegal ('>':s) = "&gt;" ++ replaceIllegal s
    replaceIllegal (x:s) = x : replaceIllegal s
    replaceIllegal [] = []

    splitBy delimiter = foldr f [[]]
          where f c l@(x:xs) | c == delimiter = []:l
                             | otherwise = (c:x):xs


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
    includeFiles :: [Char] -> IO [Char]
    includeFiles ('@':'i':'n':'c':'l':'u':'d':'e':'(':'"':t) = do
        let path = takeWhile (/= '"') t
        inc_file <- readFile path
        includeFiles $ inc_file ++ drop 1 ( dropWhile (/= ')') t)
    includeFiles ('@':'f':'r':'o':'m':'_':'c':'o':'d':'e':'(':'"':t) = do
        let path = takeWhile (/= '"') t
        inc_file <- readFile path
        let rest_path = filter (/= ' ') (takeWhile (/= ')') (drop 1 (dropWhile (/= '"') t)))
        let select_expand = filter (/= "") (splitBy ',' rest_path)
        includeFiles $ compileCppToHtml inc_file select_expand ++ drop 1 ( dropWhile (/= ')') t)
    includeFiles ('@':'g':'e':'n':'_':'t':'o':'c':'(':'"':t) = do
        let path = takeWhile (/= '"') t
        inc_file <- readFile path
        let rest_path = filter (/= ' ') (takeWhile (/= ')') (drop 1 (dropWhile (/= '"') t)))
        let select_expand = filter (/= "") (splitBy ',' rest_path)
        includeFiles $ compileTOCForCPP inc_file select_expand ++ drop 1 ( dropWhile (/= ')') t)
    includeFiles (x:t) = do
        rek <- includeFiles t
        return (x : rek)
    includeFiles [] = return []

    foldHtml :: FilePath -> IO ()
    foldHtml path = do
        content <- readFile path
        included <- includeFiles content
        writeFile (newFilename path) (transformCode included)
        where
            transformCode str = concatMap (\x -> do
                    let code = map fst x
                    if snd (head x) then
                        "<pre class=\"card code\" style=\"margin: 5px;\">"
                        ++ reverse (dropWhile (\c -> c == '\r' || c == '\n' || c == ' ') (reverse (highlightCode (highlightLiterals $ replaceIllegal code))))
                        ++ "</pre>"
                    else code)
                (groupBy (\a b -> snd a == snd b) (markCode str 0 False))
            markCode "" n _ = []
            markCode ('@':'c':'o':'d':'e':'{' :t) 0 _ = markCode t 1 True
            markCode ('{':t) n iscode = ('{', iscode) : markCode t (n + 1) iscode
            markCode ('}':t) 1 True = markCode t 0 False
            markCode ('}':t) n iscode = ('}',iscode) : markCode t (n - 1) iscode
            markCode (x : t) n iscode = (x, iscode) : markCode t n iscode

            highlightCode [] = []
            highlightCode ('c':'l':'a':'s':'s':' ':t) =
                    "<span style=\"color: " ++ highKeyword ++ "\">class</span> " ++ highlightCode t
            highlightCode ('s':'t':'r':'u':'c':'t':' ':t) =
                    "<span style=\"color: " ++ highKeyword ++ "\">struct</span> " ++ highlightCode t
            highlightCode ('i':'n':'t':t) =
                if head t `elem` ['&', ',', '.', ' '] then
                    "<span style=\"color: " ++ highType ++ "\">int</span>" ++ highlightCode t
                else "int" ++ highlightCode t
            highlightCode ('f':'l':'o':'a':'t':t) =
                if head t `elem` ['&', ',', '.', ' '] then
                    "<span style=\"color: " ++ highType ++ "\">float</span>" ++ highlightCode t
                else "float" ++ highlightCode (t)
            highlightCode ('d':'o':'u':'b':'l':'e':t) =
                if head t `elem` ['&', ',', '.', ' '] then
                    "<span style=\"color: " ++ highType ++ "\">double</span>" ++ highlightCode t
                else "double" ++ highlightCode (t)
            highlightCode ('l':'o':'n':'g':t) =
                if head t `elem` ['&', ',', '.', ' '] then
                    "<span style=\"color: " ++ highType ++ "\">long</span>" ++ highlightCode t
                else "long" ++ highlightCode (t)
            highlightCode ('s':'i':'z':'e':'_':'t':t) =
                if head t `elem` ['&', ',', '.', ' '] then
                    "<span style=\"color: " ++ highType ++ "\">double</span>" ++ highlightCode t
                else "size_t" ++ highlightCode (t)
            highlightCode ('T':'e':'n':'s':'o':'r':t) =
                if head t `elem` ['&', ',', '.', ' '] then
                    "<span style=\"color: " ++ highType ++ "\">Tensor</span>" ++ highlightCode t
                else "Tensor" ++ highlightCode (t)
            highlightCode ('/':'/':t) =
                    "<span style=\"color: #D0D0D0\">//" ++ takeWhile (\c -> c /= '\n' && c /= '\r') t ++ "</span>" ++ highlightCode (dropWhile (\c -> c /= '\n' && c /= '\r') t)
            highlightCode ('/':'*':t) = do
                    -- char and following char zipped together to detect /*
                    let interleaved = zip t (drop 1 t)
                    let comment = map fst (takeWhile (\x -> fst x /= '*' || snd x /= '/') interleaved)
                    let rest = drop 2 (map fst (dropWhile (\x -> fst x /= '*' || snd x /= '/') interleaved))
                    "<span style=\"color: #D0D0D0\">/*" ++ comment ++
                        "*/</span>" ++ highlightCode rest
            highlightCode (x:t)= x : highlightCode t

            newFilename x = take (length x - 5) x  ++ "html"
