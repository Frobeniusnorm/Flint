{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

{-# HLINT ignore "Redundant bracket" #-}
module CPPParser where

import Data.List (foldl', isInfixOf, isPrefixOf, sortOn)
import Data.Text (pack, replace, splitOn, unpack)
import Debug.Trace

-- A documentation is either a symbol with documentation or a structure with a symbol and documentation and a list of sub-documentations
data DeclDoc = SymDoc {name :: String, docu :: String} | StructDoc {name :: String, docu :: String, children :: [DeclDoc]} deriving (Show)

highlightName decl =
  highlightHelper decl "" 0
  where
    highlightHelper ('(' : t) prev 0 = "<b>" ++ prev ++ "</b>(" ++ t
    highlightHelper ('<' : '(' : t) prev 0 = "<b>" ++ prev ++ "&lt;</b>(" ++ t
    highlightHelper ('<' : '<' : '(' : t) prev 0 = "<b>" ++ prev ++ "&lt;&lt;</b>(" ++ t
    highlightHelper ('>' : '(' : t) prev 0 = "<b>" ++ prev ++ "&gt;</b>(" ++ t
    highlightHelper ('>' : '=' : t) prev nb = highlightHelper t (prev ++ "&gt;=") nb
    highlightHelper ('<' : t) prev nb = highlightHelper t (prev ++ "&lt;") (nb + 1)
    highlightHelper ('>' : t) prev nb = highlightHelper t (prev ++ "&gt;") (nb - 1)
    highlightHelper (' ' : t) prev nb = prev ++ " " ++ (highlightHelper t "" nb)
    highlightHelper s@(x : t) prev nb = do
      if "class " `isPrefixOf` s
        then "class <b>" ++ takeWhile (/= '<') (drop 6 s) ++ "</b>"
        else
          if "struct " `isPrefixOf` s
            then "struct <b>" ++ takeWhile (/= '<') (drop 7 s) ++ "</b>"
            else
              if "enum " `isPrefixOf` s
                then "enum <b>" ++ takeWhile (/= '<') (drop 5 s) ++ "</b>"
                else highlightHelper t (prev ++ [x]) nb
    highlightHelper [] prev nb = prev

-- parses a cpp file to a list of documentations and function declerations
parseCpp :: String -> [DeclDoc]
parseCpp = parseHelper
  where
    parseHelper ('/' : '*' : '*' : l) = do
      let doc = map fst (takeWhile (\a -> fst a /= '*' || snd a /= '/') (zip l (drop 1 l)))
      let afterDoc = (dropWhile (\a -> a == ' ' || a == '\t' || a == '\n' || a == '\r') (drop (length doc + 2) l))
      let func = takeWhile (\a -> a /= ';' && a /= '{') afterDoc
      let r = drop (length func) afterDoc
      if head r == '{'
        then do
          let contained = selectBlock (tail r) 0
          let rek = parseHelper contained
          let rest = drop (1 + length contained) r
          if not (null rek)
            then StructDoc func doc rek : parseHelper rest
            else SymDoc func doc : parseHelper rest
        else SymDoc func doc : parseHelper r
    parseHelper (x : t) = parseHelper t
    parseHelper [] = []
    selectBlock [] n = []
    selectBlock ('}' : r) 0 = []
    selectBlock ('{' : r) n = '{' : (selectBlock r (n + 1))
    selectBlock ('}' : r) n = '}' : (selectBlock r (n - 1))
    selectBlock (x : r) n = x : (selectBlock r n)

removeIllegal ('<' : t) = "&lt;" ++ removeIllegal t
removeIllegal ('>' : t) = "&gt;" ++ removeIllegal t
removeIllegal (x : t) = x : removeIllegal t
removeIllegal [] = []

bulletpointHighlight ('\r' : '\n' : '\r' : '\n' : str) = "<div style=\"display:block; height: 0.5em\"></div>" ++ bulletpointHighlight ('\n' : str)
bulletpointHighlight ('\n' : '\n' : str) = "<div style=\"display:block; height: 0.5em\"></div>" ++ bulletpointHighlight ('\n' : str)
bulletpointHighlight ('\n' : ' ' : '-' : str) = do
  let (subpoints, rest) = highlightpoints ('\n' : ' ' : '-' : str)
  "<ul>" ++ subpoints ++ "</ul>" ++ bulletpointHighlight rest
  where
    highlightpoints ('\n' : ' ' : '-' : str) = do
      let point =
            map
              (\(a, b, c) -> b)
              ( takeWhile
                  ( \(a, b, c) ->
                      (a /= '\n' || (b == '\t' || b == ' ') && (c == '\t' || c == ' '))
                        && not (a == '\n' && (b == ' ' || b == '\t') && c == '-')
                  )
                  (zip3 str (drop 1 str) (drop 2 str))
              )
      let (rek, rest) = highlightpoints (drop (length point) str)
      ("<li>" ++ removeIllegal point ++ "</li>" ++ rek, rest)
    highlightpoints [] = ([], [])
    highlightpoints (a : x) = ([], x)
bulletpointHighlight (x : str) = removeIllegal [x] ++ bulletpointHighlight str
bulletpointHighlight [] = []

inlineCode ('`' : t) = "<pre class=\"inline_code\">" ++ takeWhile (/= '`') t ++ "</pre>" ++ inlineCode (drop 1 (dropWhile (/= '`') t))
inlineCode (x : t) = x : inlineCode t
inlineCode [] = []

selectFcts dd sel =
  -- "." enables that all top level definitions are included
  (functionCall dd sel ("." `elem` sel), selectHelper dd sel ("." `elem` sel))
  where
    selectHelper sd@(SymDoc name doc) selection all =
      [([], sd) | (stripFctname name) `elem` selection || null selection || all]
    selectHelper sd@(StructDoc name doc c) selection all = do
      let contains = (stripFctname name) `elem` selection
      if contains || null selection || all
        then (([], sd) : (concatMap (\x -> map (\(p1, p2) -> (name : p1, p2)) (selectHelper x (selection) contains)) c))
        else []
    functionCall sd@(SymDoc name doc) selection all = do
      let contains = (stripFctname name) `elem` selection
      if contains || null selection || all
        then do
          [parseName name]
        else []
    functionCall sd@(StructDoc name doc children) selection all = do
      let contains = (stripFctname name) `elem` selection
      let rek = concatMap (\c -> functionCall c selection all) children
      if contains || null selection || all
        then do
          ((parseName name) : rek)
        else rek
    parseName name = do
      let has_class = "class " `isInfixOf` name
      let has_struct = "struct " `isInfixOf` name
      let name_part =
            if has_class || has_struct
              then takeWhile (/= '<') (unpack $ last (if has_class then splitOn (pack "class ") (pack name) else splitOn (pack "struct ") (pack name)))
              else name

      let before_ending = dropWhile (/= '(') (reverse name_part)
      let to_name = if null before_ending then reverse name_part else drop 1 before_ending
      let to_ending = reverse $ takeWhile (/= ' ') (dropWhile (== ' ') to_name)
      if head to_ending == '*' then drop 1 to_ending else to_ending

functionNameHighlighting str fn_names =
  unpack
    ( foldl
        ( \curr (fn_name, alias) ->
            ( replace (pack $ '`' : alias ++ ['`']) (pack $ " <a href=\"#" ++ fn_name ++ "\">`" ++ alias ++ "`</a>") curr
            )
        )
        (pack str)
        fn_names
    )

highlightDoc str fn_names fn_calls = do
  -- modify function names so that if the name is unique no fancy characters are needed (the original name is used as an alias)
  let alias_names = zipWith (\ n c
        -> (if length (filter (c ==) fn_calls) == 1 then
                (n, c)
            else
                (n, n))) fn_names fn_calls

  let fct_names = functionNameHighlighting (bulletpointHighlight str) alias_names
  inlineCode fct_names

stripFctname str = do
  let foo = takeWhile (\x -> x /= '{' && x /= ';') (drop 1 $ dropWhile (/= ' ') str)
  "s-"
    ++ replaceIllegal
      ( if not (null foo) && head foo == '*'
          then drop 1 foo
          else foo
      )
  where
    replaceIllegal [] = []
    replaceIllegal ('\r' : x) = replaceIllegal x
    replaceIllegal ('\n' : x) = replaceIllegal x
    replaceIllegal ('<' : x) = '_' : replaceIllegal x
    replaceIllegal ('>' : x) = '_' : replaceIllegal x
    replaceIllegal (' ' : x) = '_' : replaceIllegal x
    replaceIllegal (',' : x) = '_' : replaceIllegal x
    replaceIllegal ('.' : x) = '_' : replaceIllegal x
    replaceIllegal ('(' : x) = '_' : replaceIllegal x
    replaceIllegal (')' : x) = '_' : replaceIllegal x
    replaceIllegal (a : x) = a : replaceIllegal x

compileTOCForCPP str selection = do
  let fcts_tree = parseCpp str
  let fcts_sel = map (`selectFcts` selection) fcts_tree
  let fcts_hyra = concatMap snd fcts_sel
  let fcts_defs = map (snd) fcts_hyra
  let map_sym_html (b, a) = (concat ["&nbsp;" | i <- [0 .. (length b)]]) ++ "&#x2022;&nbsp;<a href=\"#" ++ stripFctname (name a) ++ "\">" ++ (highlightName (name a)) ++ "</a><br/>"
  let ovw_fcts = concatMap map_sym_html fcts_hyra
  "<div class=\"card\">"
    ++ "    <span class=\"card_header\">Overview</span>"
    ++ "</div><br /><div class=\"card\">\
       \<span class=\"card_header\" style=\"font-size:1.2em\">Types and Functions</span><div class=\"spacer\" style=\"height:1em\"></div>"
    ++ ovw_fcts
    ++ "<br/></div>"
  where
    isDataNode :: [Char] -> Bool
    isDataNode x = "struct " `isInfixOf` x || "enum " `isInfixOf` x || "class " `isInfixOf` x

compileCppToHtml str selection = do
  let fcts_tree = parseCpp str
  let fcts_sel = map (`selectFcts` selection) fcts_tree
  let fcts_hyra = concatMap snd fcts_sel
  let fcts_defs = map (snd) fcts_hyra
  let fcts_alias = concatMap fst fcts_sel
  let fcts_srt = sortOn (\(a, _) -> -length a) ((map (stripFctname . name) fcts_defs) `zip` fcts_alias)
  let fcts_names = map (fst) fcts_srt
  let fcts_calls = map (snd) fcts_srt
  concatMap
    ( \(b, a) ->
        "<div id=\""
          ++ stripFctname (name a)
          ++ "\"></div><div style=\"margin-left: "
          ++ show (length b)
          ++ "em;\" class=\"card\"><pre class=\"card_header_code\">"
          ++ highlightName (name a)
          ++ "</pre></div>\n<br />\n<div style=\"margin-left: "
          ++ show (length b)
          ++ "em;\" class=\"card\"><div style=\"padding: 5px;\">"
          ++ highlightDoc (parseDoc (docu a) "") fcts_names fcts_calls
          ++ "</div></div><div style=\"display: block; height: 2em;\"></div>\n"
    )
    fcts_hyra
  where
    parseDoc ('\n' : t) res = do
      let stripped = dropWhile (\x -> x == ' ' || x == '\t') t
      parseDoc (if not (null stripped) && head stripped == '*' then drop 1 stripped else stripped) (res ++ "\n")
    parseDoc (x : t) res = parseDoc t (res ++ [x])
    parseDoc [] res = res
