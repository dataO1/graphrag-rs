use clap::{Arg, Command};
use jsonfixer::{JsonRepairOptions, from_file, loads};
use std::error::Error;
use std::fs;
use std::io::{self, Read};

fn main() -> Result<(), Box<dyn Error>> {
    let matches = Command::new("jsonfixer")
        .version("0.1.0")
        .about("Repair and parse JSON files")
        .arg(
            Arg::new("input")
                .help("The JSON file to repair (if omitted, reads from stdin)")
                .index(1),
        )
        .arg(
            Arg::new("inline")
                .short('i')
                .long("inline")
                .help("Replace the file inline instead of returning the output to stdout"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("TARGET")
                .help("Write the output to TARGET filename instead of stdout"),
        )
        .arg(
            Arg::new("indent")
                .long("indent")
                .value_name("INDENT")
                .help("Number of spaces for indentation (default 2)"),
        )
        .get_matches();

    let mut options = JsonRepairOptions::default();
    if let Some(indent_str) = matches.get_one::<String>("indent") {
        options.indent = Some(indent_str.parse()?);
    }

    let result = if let Some(input_file) = matches.get_one::<String>("input") {
        from_file(input_file, options)?
    } else {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input)?;
        loads(&input, options)?
    };

    let output = serde_json::to_string_pretty(&result)?;

    if *matches.get_one("inline").unwrap_or(&false) {
        if let Some(input_file) = matches.get_one::<String>("input") {
            fs::write(input_file, &output)?;
        } else {
            return Err("Cannot use --inline with stdin".into());
        }
    } else if let Some(output_file) = matches.get_one::<String>("output") {
        fs::write(output_file, &output)?;
    } else {
        println!("{}", output);
    }

    Ok(())
}
