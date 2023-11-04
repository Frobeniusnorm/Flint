import System.Directory.Internal.Prelude (getArgs)
import System.Directory
import Data.List
import Control.Monad (forM)
import HtmlParser (foldHtml)

main = do
    args <- getArgs
    let path = head args
    files <- getDirectoryContents path
    let filtered = map (path ++) (filter (isSuffixOf ".ahtml") files)
    mapM_ foldHtml filtered
