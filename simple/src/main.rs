//use crate::data::Data;

mod data;
mod compute;


use crate::data::{read_data, Data};

fn main() {
    let data_option = read_data();
    let data = match data_option {
        None => {println!("Couldn't retrieve data!"); return},
        Some(k) => k
    };
}




