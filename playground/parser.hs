import System.IO
import Data.List
import Data.Bifunctor


-- | Custom split function
splitOnCustom :: String -> Char -> [Char] -> [String]
splitOnCustom string delim ignore = splitHelper string delim ignore [] []
    where
        splitHelper :: String -> Char -> [Char] -> String -> [String] -> [String]
        splitHelper [] delim ignore [] accum = accum --return accum
        splitHelper [] delim ignore buf accum = accum ++ [buf] --append final buf and return
        splitHelper (x:xs) delim ignore [] accum
            | x `elem` ignore || x == delim = splitHelper xs delim ignore [] accum --ignore
            | otherwise = splitHelper xs delim ignore [x] accum --build buf
        splitHelper (x:xs) delim ignore buf accum
            | x `elem` ignore = splitHelper xs delim ignore buf accum --ignore
            | x == delim = splitHelper xs delim ignore [] (accum ++ [buf]) --split
            | otherwise = splitHelper xs delim ignore (buf ++ [x]) accum --build buf


-- | Group trial data together
groupTrials :: [[String]] -> String -> [[String]]
groupTrials [] [] = []
groupTrials lis line = do
    let rev = reverse $ splitOnCustom line ' ' [] --split the line into reversed tokens
    groupHelper lis rev
        where
            -- | Helper for groupTrials
            groupHelper :: [[String]] -> [String] -> [[String]]
            groupHelper [] (num:"Trial":cul_name) = [[intercalate "_" $ reverse cul_name]] --first group
            groupHelper [] _ = []
            groupHelper ((y:ys):xs) (num:"Trial":cul_name) = [intercalate "_" $ reverse cul_name]:((y:ys):xs) --new group
            groupHelper ((y:ys):xs) (z:zs) = (z:y:ys):xs --append to existing group
            groupHelper ((y:ys):xs) [] = (y:ys):xs


-- | Number all of the seasons
numberSeason :: [String] -> (String, [(String, Integer)])
numberSeason (x:xs) = (x, numberHelper xs 0)
    where
        numberHelper :: [String] -> Integer -> [(String, Integer)]
        numberHelper [] _ = []
        numberHelper (x:xs) num = (x, num) : numberHelper xs (num + 1)


-- | Combines seasons
combineSeasons :: [(String, [[Integer]])] -> (String, [Integer]) -> [(String, [[Integer]])]
combineSeasons [] (name, lis) = [(name, [lis])]
combineSeasons ((name, lis):xs) (name1, lis1)
    | name == name1 = (name, lis1:lis):xs
    | otherwise = (name1, [lis1]):(name, reverse lis):xs


-- | Converts parsed data to printable form
printableCultivars :: String -> (String, [[Integer]]) -> String
printableCultivars str (name, seasons) = str ++ (name ++ "\n" ++ printHelper seasons ++ "\n")
    where
        -- | Converts one cultivar's trials to a string
        printHelper :: [[Integer]] -> String
        printHelper [] = []
        printHelper trials = concatMap (\x -> helperHelper x ++ "\n") trials
            where
                -- | Converts one list of integers to a string
                helperHelper :: [Integer] -> String
                helperHelper [] = []
                helperHelper season = foldl (\accum x -> accum ++ (show x ++ " ")) [] season


-- | Main entry point
main :: IO ()
main = do
    handle <- openFile "inputs/seasons.txt" ReadMode
    contents <- hGetContents handle
    let splitted = splitOnCustom contents '\n' "\r"
    let grouped_temp = reverse $ foldl groupTrials [] splitted
    let grouped = map reverse grouped_temp
    let numbered = map numberSeason grouped
    let sorted = map (Data.Bifunctor.second sort ) numbered --maps onto the second element of each tuple
    let clipped = map (second (map snd)) sorted --converts second element to just its second element (num)
    let finished = reverse $ foldl combineSeasons [] clipped
    print finished 
    let printable = foldl printableCultivars [] finished
    writeFile "inputs/seasons_parsed.txt" printable

