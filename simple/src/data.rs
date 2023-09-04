use std::{env, fs::File, io::Read, mem::{transmute, size_of}};

/*
    state_vectors = np.zeros((32, 252, 2048), dtype = np.float32)
    vectors = np.zeros((32, 252, 1024), dtype = np.float32)
    lte = np.zeros((32, 3, 252), dtype = np.float32)
*/


/// Holds all data for easy management.
pub struct Data {
    //state_vectors: [[[f32; 32]; 252]; 2048],
    //vectors: [[[f32; 32]; 252]; 1024],
    //lte: [[[f32; 32]; 3]; 252],
    pub state_vectors: Vec<f32>, // 32 * 252 * 2048
    pub vectors: Vec<f32>,       // 32 * 252 * 1024
    pub lte: Vec<f32>,           // 32 * 3 * 252
}
impl Data {
    //pub fn new() -> Self {
    //    Data {
    //        //state_vectors: vec![vec![vec![0.0; 2048]; 252]; 32],
    //        //vectors: vec![vec![vec![0.0; 2048]; 252]; 32],
    //        //lte: vec![vec![vec![0.0; 2048]; 252]; 32],
    //    }
    //}
}


/// Read data from .data files produced by 'playground'
pub fn read_data() -> Option<Data> {
    let cur_dir = env::current_dir().ok()?.to_str()?.to_owned();
    let data_loc = cur_dir + "/../playground";
    //println!("{:?}", data_loc);

    const STATE_VECTORS_SIZE: usize = 32 * 252 * 2048 * 4;
    const VECTORS_SIZE: usize = 32 * 252 * 1024 * 4;
    const LTE_SIZE: usize = 32 * 3 * 252 * 4;

    //let mut st_buf = Vec::with_capacity(STATE_VECTORS_SIZE);
    //let mut vc_buf = Vec::with_capacity(VECTORS_SIZE);
    //let mut lt_buf = Vec::with_capacity(LTE_SIZE);
    //let mut sv_buf = [0; STATE_VECTORS_SIZE];
    //let mut st_buf = [0; STATE_VECTORS_SIZE];
    //let mut vc_buf = [0; VECTORS_SIZE];
    //let mut lt_buf = [0; LTE_SIZE];

    //println!("{}", STATE_VECTORS_SIZE);
    //println!("{}", size_of::<u8>());
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

    //let x = f32::from_le_bytes(st_buf);
    //let state_vectors: Vec::<Option<f32>> = st_buf
    //    .chunks(4)
    //    .map(|chunk| 
    //        f32::from_le_bytes(chunk)
    //    )
    //    .collect();
    //st_buf.get_f32_le();
    //let state_vectors = unsafe {st_buf.align_to::<f32>()};
    //let vectors = unsafe {st_buf.align_to::<f32>()};
    //let lte = unsafe {st_buf.align_to::<f32>()};
    //println!("hi");
    ////let state_vectors: Vec<f32> = (unsafe {transmute::<&[u8], &[f32]>(&st_buf)}).to_vec();
    //let state_vectors = unsafe {transmute::<&[u8], &[f32]>(&st_buf)};
    //println!("ha");
    //println!("{}", state_vectors.len());
    //let vectors: Vec<f32> = (unsafe {transmute::<&[u8], &[f32]>(&vc_buf)}).to_vec();
    //let lte: Vec<f32> = (unsafe {transmute::<&[u8], &[f32]>(&lt_buf)}).to_vec();

    //println!("st_buf size: {}", st_buf.len());
    //println!("state_vectors size: {}", state_vectors.len());
    //println!("vectors size: {}", vectors.len());
    //println!("lte size: {}", lte.len());
    let state_vectors: Vec<f32> = st_buf.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();
    let vectors: Vec<f32> =       vc_buf.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();
    let lte: Vec<f32> =           lt_buf.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();

    //println!("state_vectors size: {}", state_vectors.len());
    //println!("vectors size: {}", vectors.len());
    //println!("lte size: {}", lte.len());

    Some(Data {state_vectors, vectors, lte})

    //Some(Data::new())
}



