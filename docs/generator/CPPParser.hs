{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Redundant bracket" #-}
module CPPParser where
    import Data.List (isPrefixOf, foldl', sortOn, isInfixOf)
    import Data.Text (replace, unpack, pack)
    -- A documentation is either a symbol with documentation or a structure with a symbol and documentation and a list of sub-documentations
    data DeclDoc = SymDoc {name::String, docu::String} | StructDoc {name::String, docu::String, children::[DeclDoc]}
    -- parses a cpp file to a list of documentations and function declerations
    parseCpp :: String -> [DeclDoc]
    parseCpp = parseHelper
        where
            parseHelper ('/':'*':'*':l) = do
                let doc = map fst (takeWhile (\a -> fst a /= '*' || snd a /= '/') (zip l (drop 1 l)))
                let afterDoc = (dropWhile (\a -> a == ' ' || a == '\t' || a == '\n' || a == '\r') (drop (length doc + 2) l))
                let func = takeWhile (\a -> a /= ';' && a /= '{') afterDoc
                let r = drop (length func) afterDoc
                if head r == '{' then
                    do
                        let contained = selectBlock (tail r) 0
                        let rek = parseHelper contained
                        let rest = drop (1 + length contained) r
                        if not (null rek) then
                            StructDoc func doc rek  : parseHelper rest
                        else
                            SymDoc func doc  : parseHelper rest
                else
                    SymDoc func doc : parseHelper r
            parseHelper (x:t) = parseHelper t
            parseHelper [] = []
            selectBlock [] n = []
            selectBlock ('}':r) 0 = []
            selectBlock ('{':r) n = '{':(selectBlock r (n + 1))
            selectBlock ('}':r) n = '}':(selectBlock r (n - 1))
            selectBlock (x:r) n = x:(selectBlock r n)

    removeIllegal ('<':t) = "&lt;" ++ removeIllegal t
    removeIllegal ('>':t) = "&gt;" ++ removeIllegal t
    removeIllegal (x:t) = x : removeIllegal t
    removeIllegal [] = []

    bulletpointHighlight ('\r':'\n':'\r':'\n':str) = "<div style=\"display:block; height: 0.5em\"></div>" ++ bulletpointHighlight ('\n':str)
    bulletpointHighlight ('\n':' ':'-':str) = do
        let (subpoints, rest) = highlightpoints ('\n':' ':'-':str)
        "<ul>" ++ subpoints ++ "</ul>" ++ bulletpointHighlight rest
        where
            highlightpoints ('\n':' ':'-':str) = do
                let point = map (\(a,b,c) -> b) (takeWhile (\(a,b,c) -> (a /= '\n' || (b == '\t' || b == ' ') && (c == '\t' || c == ' ')) &&
                                                        not (a == '\n' && (b == ' ' || b == '\t') && c == '-'))
                                                        (zip3 str (drop 1 str) (drop 2 str)))
                let (rek, rest) = highlightpoints (drop (length point) str)
                ("<li>" ++ removeIllegal point ++ "</li>" ++ rek, rest)
            highlightpoints [] = ([], [])
            highlightpoints (a:x) = ([], x)
    bulletpointHighlight (x:str) = removeIllegal [x] ++ bulletpointHighlight str
    bulletpointHighlight [] = []


    inlineCode ('`':t) = "<pre class=\"inline_code\">" ++ takeWhile (/= '`') t ++ "</pre>" ++ inlineCode (drop 1 (dropWhile (/= '`') t))
    inlineCode (x:t) = x:inlineCode t
    inlineCode [] = []

    functionNameHighlighting str fn_names =
        unpack (foldl (\curr fn_name ->
            replace (pack $ '`':fn_name ++ ['`']) (pack $ " <a href=\"#" ++ fn_name ++ "\">`" ++ fn_name ++ "`</a>") curr)
            (pack str) fn_names)
    highlightDoc str fn_names= inlineCode (functionNameHighlighting (bulletpointHighlight str) fn_names)
    stripFctname str = do
                let foo = takeWhile (\x -> x /= '(' && x /= '{' && x /= ';') (drop 1 $ dropWhile (/= ' ') str)
                replaceIllegal $
                    if not (null foo) && head foo == '*' then
                        drop 1 foo else foo
        where
            replaceIllegal [] = []
            replaceIllegal ('<':x) = '_' : replaceIllegal x
            replaceIllegal ('>':x) = '_' : replaceIllegal x
            replaceIllegal (' ':x) = '_' : replaceIllegal x
            replaceIllegal (',':x) = '_' : replaceIllegal x
            replaceIllegal (a:x) = a : replaceIllegal x

    compileTOCForCPP str = do
        let fcts_defs = parseCpp str
        let ovw_fcts = concatMap (\a -> "<li><a href=\"#" ++ stripFctname (name a) ++ "\">" ++ removeIllegal (name a) ++ "</a></li>")
                (filter (\x -> not (isDataNode $ name x)) fcts_defs)
        let ovw_types = concatMap (\a -> "<li><a href=\"#" ++ stripFctname (name a) ++ "\">" ++ removeIllegal (name a) ++ "</a></li>")
                (filter (isDataNode . name) fcts_defs)
        "<div class=\"card\">" ++
            "    <span class=\"card_header\">Overview</span>" ++
            "</div><br /><div class=\"card\">\
            \<span class=\"card_header\" style=\"font-size:1.2em\">Types</span><ul>"
            ++ ovw_types ++
            "</ul>\
            \<span class=\"card_header\" style=\"font-size:1.2em\">Functions</span><ul>"
            ++ ovw_fcts ++
            "</ul></div>"
        where
            isDataNode::[Char] -> Bool
            isDataNode x = "struct " `isInfixOf` x || "enum " `isInfixOf` x || "class " `isInfixOf` x

    compileCppToHtml str = do
        let fcts_defs = parseCpp str
        let fct_names = sortOn (\a -> -length a) (map (stripFctname . name) fcts_defs)
        concatMap (\a ->
            "<div id=\""
                ++ stripFctname (name a) ++
                "\"></div><div class=\"card\"><pre class=\"card_header_code\">"
                ++ removeIllegal (name a) ++
                "</pre></div>\n<br />\n<div class=\"card\"><div style=\"padding: 5px;\">"
                ++ highlightDoc (parseDoc (docu a) "") fct_names ++
                "</div></div><div style=\"display: block; height: 2em;\"></div>\n") fcts_defs
        where
            parseDoc ('\n':t) res = do
                let stripped = dropWhile (\x -> x == ' ' || x == '\t') t
                parseDoc (if not (null stripped) && head stripped == '*' then drop 1 stripped else stripped) (res ++ "\n")
            parseDoc (x:t) res = parseDoc t (res ++ [x])
            parseDoc [] res = res