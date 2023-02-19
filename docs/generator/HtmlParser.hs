module HtmlParser where
    highlightKeyword = "#F030FF"
    highlightType = "#FFF030"

    replaceIllegal ('<':s) = "&lt;" ++ replaceIllegal s
    replaceIllegal ('>':s) = "&gt;" ++ replaceIllegal s
    replaceIllegal (x:s) = x : replaceIllegal s
    replaceIllegal [] = []

    foldHtml :: FilePath -> IO ()
    foldHtml path = do
        content <- readFile path
        writeFile (newFilename path) (highlightCode content 0)
        where
            highlightCode "" n = []
            highlightCode ('@':'c':'o':'d':'e':'{' :t) 0 = highlightCode t 1
            highlightCode ('{':t) n = '{' : highlightCode t (n + 1)
            highlightCode ('}':t) 0 = '}' : highlightCode t 0
            highlightCode ('}':t) n = '}' : highlightCode t (n - 1)
            highlightCode ('c':'l':'a':'s':'s':t) n = if n > 0 then
                    "<span style=\"color: " ++ highlightKeyword ++ "\">class</span>" ++ highlightCode t n
                else "class" ++ highlightCode t n
            highlightCode ('i':'n':'t':t) n = if n > 0 then
                    "<span style=\"color: " ++ highlightType ++ "\">int</span>" ++ highlightCode t n
                else "int" ++ highlightCode t n
            highlightCode ('T':'e':'n':'s':'o':'r':t) n = if n > 0 then
                    "<span style=\"color: " ++ highlightType ++ "\">Tensor</span>" ++ highlightCode t n
                else "Tensor" ++ highlightCode t n
            highlightCode ('/':'/':t) n = if n > 0 then
                    "<span style=\"color: #D0D0D0\">" ++ takeWhile (/= '\n') t ++ "</span>" ++ highlightCode (dropWhile (/= '\n') t) n
                else "//" ++ highlightCode t n
            highlightCode ('/':'*':t) n = if n > 0 then do
                    -- char and following char zipped together to detect /*
                    let interleaved = zip t (drop 1 t)
                    let comment = replaceIllegal $ map fst (takeWhile (\x -> fst x /= '*' || snd x /= '/') interleaved)
                    let rest = drop 2 (map fst (dropWhile (\x -> fst x /= '*' || snd x /= '/') interleaved))
                    "<span style=\"color: #D0D0D0\">/*" ++ comment ++
                        "*/</span>" ++ highlightCode rest n
                else "//" ++ highlightCode t n
            -- here the replacement for < > happens
            highlightCode (x:t) n = (if n > 0 then replaceIllegal [x] else [x]) ++ highlightCode t n
            newFilename x = take (length x - 5) x  ++ "html"