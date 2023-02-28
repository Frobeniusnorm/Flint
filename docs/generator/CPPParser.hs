module CPPParser where
    import Data.List (isPrefixOf, foldl', sortOn)
    import Data.Text (replace, unpack, pack)
    -- parses a cpp file to a list of documentations and function declerations
    parseCpp :: String -> [(String, String)]
    parseCpp = parseHelper
        where
            parseHelper ('/':'*':'*':l) = do
                let doc = map fst (takeWhile (\a -> fst a /= '*' || snd a /= '/') (zip l (drop 1 l)))
                let func = takeWhile (\a -> a /= ';' && a /= '{')
                        (dropWhile (\a -> a == ' ' || a == '\t' || a == '\n' || a == '\r') (drop (length doc + 2) l))
                let r = drop (length doc + length func) l
                (doc, func) : parseHelper r
            parseHelper (x:t) = parseHelper t
            parseHelper [] = []

    removeIllegal ('<':t) = "&lt;" ++ removeIllegal t 
    removeIllegal ('>':t) = "&gt;" ++ removeIllegal t
    removeIllegal (x:t) = x : removeIllegal t  
    removeIllegal [] = []

    bulletpointHighlight ('\n':'\n':str) = "<br />" ++ bulletpointHighlight ('\n':str)
    bulletpointHighlight ('\n':' ':'-':str) = do
        let (subpoints, rest) = highlightpoints ('\n':' ':'-':str)
        "<ul>" ++ subpoints ++ "</ul>" ++ bulletpointHighlight rest
        where
            highlightpoints ('\n':' ':'-':str) = do
                let point = map (\(a,b,c) -> b) (takeWhile (\(a,b,c) -> (a /= '\n' || ((b == '\t' || b == ' ') && (c == '\t' || c == ' '))) &&
                                                        not (a == '\n' && (b == ' ' || b == '\t') && c == '-'))
                                                        (zip3 str (drop 1 str) (drop 2 str)))
                let (rek, rest) = highlightpoints (drop (length point) str)
                ("<li>" ++ removeIllegal point ++ "</li>" ++ rek, rest)
            highlightpoints [] = ([], [])
            highlightpoints (a:x) = ([], x)
    bulletpointHighlight (x:str) = removeIllegal [x] ++ bulletpointHighlight str
    bulletpointHighlight [] = []

    removeLinebreak ('\n':t) = removeLinebreak t
    removeLinebreak ('\r':t) = removeLinebreak t
    removeLinebreak (x:t) = x:removeLinebreak t
    removeLinebreak [] = []

    inlineCode ('`':t) = "<pre class=\"inline_code\">" ++ removeLinebreak (takeWhile (/= '`') t) ++ "</pre>" ++ inlineCode (drop 1 (dropWhile (/= '`') t))
    inlineCode (x:t) = x:inlineCode t
    inlineCode [] = []

    functionNameHighlighting :: String -> [String] -> String
    functionNameHighlighting str fn_names =
        unpack (foldl (\curr fn_name -> 
            replace (pack $ '`':fn_name ++ ['`']) (pack $ " <a href=\"#" ++ fn_name ++ "\">`" ++ fn_name ++ "`</a>") curr) 
            (pack str) fn_names)
    highlightDoc str fn_names= inlineCode (functionNameHighlighting (bulletpointHighlight str) fn_names)

    compileCppToHtml str = do
        let fcts_defs = parseCpp str
        let ovw_fcts = concatMap (\a -> "<li><a href=\"#" ++ strip_fctname (snd a) ++ "\">" ++ snd a ++ "</a></li>")
                (filter (\a -> not ("enum" `isPrefixOf` snd a) && not ("struct" `isPrefixOf` snd a)) fcts_defs)
        let ovw_types = concatMap (\a -> "<li><a href=\"#" ++ strip_fctname (snd a) ++ "\">" ++ snd a ++ "</a></li>")
                (filter (\a -> ("enum" `isPrefixOf` snd a) || ("struct" `isPrefixOf` snd a)) fcts_defs)
        let fct_names = sortOn (\a -> -length a) (map (strip_fctname . snd) fcts_defs)
        "<div class=\"card\">" ++
            "    <span class=\"card_header\">Overview</span>" ++
            "</div><br /><div class=\"card\">\
            \<span class=\"card_header\" style=\"font-size:1.2em\">Types</span><ul>"
            ++ ovw_types ++
            "</ul>\
            \<span class=\"card_header\" style=\"font-size:1.2em\">Functions</span><ul>"
            ++ ovw_fcts ++
            "</ul></div><div style=\"display: block; height: 2em;\"></div>" ++
            concatMap (\a ->
                "<div id=\""
                    ++ strip_fctname (snd a) ++
                    "\"></div><div class=\"card\"><pre class=\"card_header_code\">"
                    ++ snd a ++
                    "</pre></div>\n<br />\n<div class=\"card\"><div style=\"padding: 5px;\">"
                    ++ highlightDoc (parseDoc (fst a) "") fct_names ++
                    "</div></div><div style=\"display: block; height: 2em;\"></div>\n") fcts_defs
        where
            parseDoc ('\n':t) res = do
                let stripped = dropWhile (\x -> x == ' ' || x == '\t') t
                parseDoc (if not (null stripped) && head stripped == '*' then drop 1 stripped else stripped) (res ++ "\n")
            parseDoc (x:t) res = parseDoc t (res ++ [x])
            parseDoc [] res = res

            strip_fctname str = do
                let foo = takeWhile (\x -> x /= '(' && x /= ' ') (drop 1 $ dropWhile (/= ' ') str)
                if not (null foo) && head foo == '*' then
                    drop 1 foo else foo