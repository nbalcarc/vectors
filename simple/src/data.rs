use std::{env, fs::File, io::Read};

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
    pub fn index_state_vectors(&self, index: usize, index1: usize, index2: usize) -> f32 {
        self.state_vectors[index*516096 + index1*2048 + index2]
    }

    pub fn index_vectors(&self, index: usize, index1: usize, index2: usize) -> f32 {
        self.state_vectors[index*258048 + index1*1024 + index2]
    }

    pub fn index_lte(&self, index: usize, index1: usize, index2: usize) -> f32 {
        self.state_vectors[index*756 + index1*252 + index2]
    }
}


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



