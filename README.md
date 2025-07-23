# Czech Judicial Decisions Scraper

This repository is second part of the SOUDAI project, which aims to predict the duration of court cases in the Czech republic. The first part of the project, [Czech Judicial Decisions Scraper](https://github.com/Mark-Bugden/SOUDAI_data_scraping), is focussed on scraping, cleaning, and preprocessing court data from publically available sources. The focus of this repository is the analysis of this data, and the development, training, and evaluation of machine-learning models which aim to predict the duration of Czech court cases. 

## Project Structure

```
data/
    processed_decisions.csv
notebooks/
references/
  court_map.yaml
src/
  training/
  config

```

## Installation

This project uses Poetry for dependency management. To install poetry, follow the instructions [here](https://python-poetry.org/docs/). 
Once poetry is installed, you can install the project as follows:

```bash
git clone https://github.com/Mark-Bugden/SOUDAI_model_training.git
cd SOUDAI_model_training
poetry install
```


## Data Folders

This repository relies on one external data file, namely the `processed_decisions.csv` file which can be obtained by running the scraping and preprocessing pipeline from [Czech Judicial Decisions Scraper](https://github.com/Mark-Bugden/SOUDAI_data_scraping). The .csv file should be copied to the data folder at `data/processed_decisions.csv` before running the notebooks in this repository. 


## Contributing

Contributions are welcome. Please submit pull requests with clear, well-documented code, and add tests where relevant.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or suggestions, please open an issue or contact the maintainer listed in the `pyproject.toml`.
