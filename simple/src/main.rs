//use crate::data::Data;

mod data;


use crate::data::{read_data, Data};

fn main() {
    println!("Hello, world!");
    
    let data_option = read_data();
    let data = match data_option {
        None => {println!("Couldn't retrieve data!"); return},
        Some(k) => k
    };
}




