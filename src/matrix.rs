use rand::Rng;
use std::fmt::Display;

#[derive(Default, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<f64>,
}

impl Display for Matrix{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result{
        let mut data_display = String::new();

        for row in 0..self.rows {
            for col in 0..self.columns {
                data_display.push_str(&format!("\t{}", self.data[row*self.columns + col].to_string()));
            } 
            data_display.push_str("\n");
        }

        write!(f, "Matrix[{},{}]\n{}", self.rows, self.columns, data_display)
    }
}

impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Matrix {
        Matrix{
            columns,
            rows,
            data: Vec::with_capacity(columns*rows),
        }
    }

    pub fn rand(rows: usize, columns: usize) -> Matrix{
        let mut buffer = Vec::<f64>::with_capacity(rows * columns);

        for _ in 0..rows*columns {
            buffer.push(rand::thread_rng().gen_range(0.0..1.0));
        }

        Matrix { rows, columns, data: buffer }
    }

    pub fn from_i32(rows: usize, columns: usize, data: i32) -> Matrix{
        let data_fill = data as f64;
        Matrix { rows, columns, data: vec![data_fill; rows*columns]}
    }

    pub fn from_string(data: &str) -> Matrix {
        let rows = data.chars().filter(|x| x == &';').count() + 1;
        let columns = data.split(";").next().unwrap().chars().filter(|x| x == &' ').count() + 1;
        
        let data = data.clone().replace(";", " ").split(" ").map(|x|  x.parse::<f64>().unwrap()).collect::<Vec<f64>>();
        Matrix { rows, columns, data}
    }

    pub fn from_buffer(rows: usize, columns: usize, data: Vec<f64>) -> Matrix {
        Matrix{ rows, columns, data: data}
    }

    pub fn from_vec(data: &Vec<f64>) -> Matrix {
        Matrix{rows: 1, columns: data.len(), data: data.clone()}
    }

    pub fn to_vec(self) -> Vec<f64> {
        let mut output = vec![];
        for index in 0..self.rows*self.columns{
            output.push(self.data[index]);
        }
        output
    }
}

impl Matrix {
    pub fn sub(&self, other: &Matrix) -> Matrix {
        let mut buffer = Vec::with_capacity(self.columns*self.rows);

        self.data
            .iter()
            .enumerate()
            .for_each(|(index, data)| buffer.push(data.clone() - other.data[index]));

        Matrix::from_buffer(self.rows, self.columns, buffer)
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        let mut buffer = Vec::with_capacity(self.columns*self.rows);

        self.data
            .iter()
            .enumerate()
            .for_each(|(index, data)| buffer.push(data.clone() + other.data[index]));

        Matrix::from_buffer(self.rows, self.columns, buffer)
    }

    pub fn multiply_dot(&self, other: &Matrix) -> Matrix {
        let mut buffer = Vec::with_capacity(self.columns*self.rows);

        self.data
            .iter()
            .enumerate()
            .for_each(|(index, data)| buffer.push(data.clone() * other.data[index]));

        Matrix::from_buffer(self.rows, self.columns, buffer)
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        let mut buffer = Vec::<f64>::with_capacity(self.rows*other.columns);

        for row in 0..self.rows {
            for column in 0..other.columns {
                let mut data: f64 = 0.0;
                for index in 0..self.columns{
                    data += self.data[row*self.columns + index] * other.data[index*other.columns + column] ;
                }
                buffer.push(data);
            }
        }
        Matrix::from_buffer(self.rows, other.columns, buffer)
    }

    pub fn transpose(&self) -> Matrix{
        Matrix { rows: self.columns, columns: self.rows, data: self.data.clone() }
    }

    pub fn map_mut(mut self, function: &dyn Fn(f64) -> f64) -> Self {
        for index in 0..self.rows*self.columns{
            self.data[index] = function(self.data[index]);
        }
        self
	}

}