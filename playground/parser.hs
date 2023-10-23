import System.IO


splitOnCustom :: String -> [String]
splitOnCustom string = splitHelper string [] []
    where
        splitHelper :: String -> String -> [String] -> [String]
        splitHelper [] [] accum = accum
        splitHelper [] buf accum = accum ++ [buf]
        splitHelper (x:xs) buf accum
            | x == '\r' = splitHelper xs buf accum
            | x == '\n' = splitHelper xs [] (accum ++ [buf])
            | otherwise = splitHelper xs (buf ++ [x]) accum


-- | Main entry point
main :: IO ()
main = do
    handle <- openFile "seasons.txt" ReadMode
    contents <- hGetContents handle
    let splitted = splitOnCustom contents
    print splitted 

