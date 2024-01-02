code and dataset for "A novel hierarchical feature selection with local shuffling and models reweighting for stock price forecasting"

Generation tutorial details at https://qlib.readthedocs.io/en/latest/component/data.html#qlib-format-data
The official qlib website is available at https://github.com/microsoft/qlib

Converting CSV Format into Qlib Format. 

Qlib has provided the script scripts/dump_bin.py to convert any data in CSV format into .bin files (Qlib format) as long as they are in the correct format.

Besides downloading the prepared demo data, users could download demo data directly from the Collector as follows for reference to the CSV format. Here are some example:

for daily data:

    python scripts/get_data.py download_data --file_name csv_data_cn.zip --target_dir ~/.qlib/csv_data/cn_data

for 1min data:

    python scripts/data_collector/yahoo/collector.py download_data --source_dir ~/.qlib/stock_data/source/cn_1min --region CN --start 2021-05-20 --end 2021-05-23 --delay 0.1 --interval 1min --limit_nums 10

Users can also provide their own data in CSV format. However, the CSV data must satisfies following criterions:

    CSV file is named after a specific stock or the CSV file includes a column of the stock name

            Name the CSV file after a stock: SH600000.csv, AAPL.csv (not case sensitive).

            CSV file includes a column of the stock name. User must specify the column name when dumping the data. Here is an example:

                python scripts/dump_bin.py dump_all ... --symbol_field_name symbol

    CSV file must includes a column for the date, and when dumping the data, user must specify the date column name. Here is an example:

    	python scripts/dump_bin.py dump_all ... --date_field_name date

Supposed that users prepare their CSV format data in the directory ~/.qlib/csv_data/my_data, they can run the following command to start the conversion.

python scripts/dump_bin.py dump_all --csv_path  ~/.qlib/csv_data/my_data --qlib_dir ~/.qlib/qlib_data/my_data --include_fields open,close,high,low,volume,factor

For other supported parameters when dumping the data into .bin file, users can refer to the information by running the following commands:

python dump_bin.py dump_all --help

After conversion, users can find their Qlib format data in the directory ~/.qlib/qlib_data/my_data.

The arguments of –include_fields should correspond with the column names of CSV files. The columns names of dataset provided by Qlib should include open, close, high, low, volume and factor at least.

    open
        The adjusted opening price

    close
        The adjusted closing price

    high
        The adjusted highest price

    low
        The adjusted lowest price

    volume
        The adjusted trading volume

    factor
        The Restoration factor. Normally, factor = adjusted_price / original_price, adjusted price reference: split adjusted

In the convention of Qlib data processing, open, close, high, low, volume, money and factor will be set to NaN if the stock is suspended. If you want to use your own alpha-factor which can’t be calculate by OCHLV, like PE, EPS and so on, you could add it to the CSV files with OHCLV together and then dump it to the Qlib format data.
