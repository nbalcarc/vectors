use std::{env, fs::File, io::Read, ops::RangeBounds};
use ndarray;

/*
    state_vectors = np.zeros((32, 252, 2048), dtype = np.float32)
    vectors = np.zeros((32, 252, 1024), dtype = np.float32)
    lte = np.zeros((32, 3, 252), dtype = np.float32)
*/


/// Holds all data for easy management.
pub struct Data {
    pub state_vectors: Vec<f32>, // 32 * 252 * 2048
    pub vectors: Vec<f32>,       // 32 * 252 * 1024
    pub lte: Vec<f32>,           // 32 * 3 * 252
}
impl Data {
    /// Returns the given index in self.state_vectors, None implies invalid indices
    pub fn index_state_vectors(&self, index: usize, index1: usize, index2: usize) -> Option<f32> {
        if index >= 32 || index1 >= 252 || index2 > 2048 {
            return None;
        }
        Some(self.state_vectors[index*516096 + index1*2048 + index2])
    }

    /// Returns the given index in self.vectors, None implies invalid indices
    pub fn index_vectors(&self, index: usize, index1: usize, index2: usize) -> Option<f32> {
        if index >= 32 || index1 >= 252 || index2 > 1024 {
            return None;
        }
        Some(self.state_vectors[index*258048 + index1*1024 + index2])
    }

    /// Returns the given index in self.lte, None implies invalid indices
    pub fn index_lte(&self, index: usize, index1: usize, index2: usize) -> Option<f32> {
        if index >= 32 || index1 >= 3 || index2 > 252 {
            return None;
        }
        Some(self.state_vectors[index*756 + index1*252 + index2])
    }
}



/// Vector with added capabilities
//pub struct MultiVector {
//    pub vec: Vec<f32>,
//    pub dimensions: Vec<usize>,
//}
//impl MultiVector {
//    /// Constructs a new, empty SmartVector
//    pub fn new(dimensions: Vec<usize>) -> Self {
//        let length = dimensions.iter().fold(0, |x, y| x*y);
//        MultiVector { vec: vec![0.0; length], dimensions }
//    }
//
//    /// Constructs a new SmartVector from an existing vector
//    pub fn from(vec: Vec<f32>, dimensions: Vec<usize>) -> Option<Self> {
//        let length = dimensions.iter().fold(0, |x, y| x*y);
//        if vec.len() != length { //quick check to ensure sizing is correct
//            return None;
//        }
//        Some(MultiVector { vec, dimensions })
//    }
//
//    /// Consumes the struct and returns its inner vector
//    pub fn flatten(self) -> Vec<f32> {
//        self.vec
//    }
//
//    /// Gets the requested index
//    pub fn index(&self, coords: Vec<usize>) -> Option<f32> {
//        if coords.len() != self.dimensions.len() { //ensure coords matches our dimensions
//            return None;
//        }
//        for i in 0..self.dimensions.len() { //ensure the coords are in bounds
//            if coords[i] > self.dimensions[i] {
//                return None;
//            }
//        }
//        let mut accumulator = 1;
//        let mut index = 0;
//        for i in (0..self.dimensions.len()).rev() {
//            index += coords[i] * accumulator;
//            accumulator *= self.dimensions[i]; //compound the target indices
//        }
//        Some(self.vec[index])
//
//    }
//
//    //pub fn extract<R: RangeBounds<usize>>(&self, dimensions: Vec<R>) {
//    //    for i in dimensions[0].start_bound()..dimensions[0].end_bound() {
//    //        println!("{}", i);
//    //    }
//    //    
//    //}
//
//    /*
//        [(0, 6), (0, 4)]
//        18, 19, 20, 21, 22, 23
//        12, 13, 14, 15, 16, 17
//        6,  7,  8,  9,  10, 11
//        0,  1,  2,  3,  4,  5
//
//        [(0, -2), (0, 4)] aka [(0, 4), (0, 4)]
//        18, 19, 20, 21
//        12, 13, 14, 15
//        6,  7,  8,  9
//        0,  1,  2,  3
//
//        [(0, -3), (0, 4)] aka [(0, 3), (0, 4)]
//        18, 19, 20
//        12, 13, 14
//        6,  7,  8
//        0,  1,  2
//
//        [(2, 3), (1, 2)]
//        14, 15,
//        8,  9, 
//        
//    */
//
//    //pub fn extract(&self, dimensions: Vec<(usize, usize)>) -> Option<MultiVector> {
//    //    if dimensions.len() != self.dimensions.len() { //ensure coords matches our dimensions
//    //        return None;
//    //    }
//    //    let mut new_dimensions = Vec::with_capacity(self.dimensions.len());
//    //    for (i, (start, end)) in dimensions.into_iter().enumerate() {
//    //        if start > 0 && end > start { //normal case, just get the range
//    //            
//    //        }
//    //    }
//    //    None
//    //}
//
//    /// Extract the given slice into another MultiVector
//    /// -1 means entire dimension
//    pub fn extract(&self, dimensions: Vec<i32>) -> Option<Self> {
//        if dimensions.len() != self.dimensions.len() {
//            return None;
//        }
//        None
//
//
//    }
//}



/// Read data from .data files produced by 'playground'
pub fn read_data() -> Option<Data> {
    let cur_dir = env::current_dir().ok()?.to_str()?.to_owned();
    let data_loc = cur_dir + "/../playground";

    const STATE_VECTORS_SIZE: usize = 32 * 252 * 2048 * 4;
    const VECTORS_SIZE: usize = 32 * 252 * 1024 * 4;
    const LTE_SIZE: usize = 32 * 3 * 252 * 4;

    let mut st_buf = vec![0; STATE_VECTORS_SIZE];
    let mut vc_buf = vec![0; VECTORS_SIZE];
    let mut lt_buf = vec![0; LTE_SIZE];

    let mut file;
    file = File::open(data_loc.clone() + "/state_vectors.data").ok()?;
    file.read_exact(&mut st_buf).ok()?; //reads file contents into buffer

    file = File::open(data_loc.clone() + "/vectors.data").ok()?;
    file.read_exact(&mut vc_buf).ok()?;

    file = File::open(data_loc + "/lte.data").ok()?;
    file.read_exact(&mut lt_buf).ok()?;

    let state_vectors: Vec<f32> = st_buf.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();
    let vectors: Vec<f32> =       vc_buf.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();
    let lte: Vec<f32> =           lt_buf.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();

    Some(Data {state_vectors, vectors, lte})
}



