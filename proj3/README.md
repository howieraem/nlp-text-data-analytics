# Project 3

## Team Members
- J Lin
- F Cao

## Files
- `INTEGRATED-DATASET.csv`: Cleaned CSV from NYPD complaint dataset
- `preprocess.py`: Script to clean up the original NYPD complaint dataset
- `example-run.txt`: Output of the compelling sample run (renamed from the produced `output.txt`)
- `requirements.txt`: Libraries that are required for our preprocessing step
- `main.py`: Where it's at!

## How to Run
`python3 main.py INTEGRATED-DATASET.csv <min_sup> <min_conf>` \
(This will produce an output.txt and only relies on the standard Python (>=3.6) packages)

If you wish to automatically convert the raw dataset to `INTEGRATED-DATASET.csv` (detailed description below), you can follow these steps:
 1. Create a new virtual environment with Python >= 3.6 and activate it
 2. Run `pip install -r requirements.txt`
 3. Download the dataset: [NYPD Complaint Data Current (Year To Date)](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Current-Year-To-Date-/5uac-w243) and add the dataset to the same directory that our code lives in
 4. Run `python3 preprocess.py`

## NYC Open Data dataset
(a) We are using the [NYPD Complaint Data Current (Year To Date)](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Current-Year-To-Date-/5uac-w243) which contains a breakdown of every valid felony, misdemeanor, and violation crimes reported to the New York City Police Department (NYPD) for all of 2021. This data is manually extracted every quarter and reviewed by the Office of Management Analysis and Planning. Each record represents a crime effected in NYC by the NYPD and includes information about the type of crime, the location and time of enforcement. In addition, information related to suspect and victim demographics is also included.

(b) The procedure we used to map the original dataset into our INTEGRATED-DATASET file is as follows:
- Read the raw dataset `NYPD_Complaint_Data_Current__Year_To_Date_.csv` with pandas as a dataframe.
- We looked through the 36 columns and picked out the columns that would be of interest. 
- We dropped the columns that we would not be using: `CMPLNT_NUM`, `ADDR_PCT_CD`, `CMPLNT_TO_DT`, `CMPLNT_TO_TM`, `CRM_ATPT_CPTD_CD`, `HADEVELOPT`, `HOUSING_PSA`, `JURISDICTION_CODE`, `JURIS_DESC`, `KY_CD`, `LAW_CAT_CD`, `PARKS_NM`, `PATROL_BORO`, `PD_CD`, `PD_DESC`, `RPT_DT`, `STATION_NAME`, `SUSP_AGE_GROUP`, `SUSP_RACE`, `SUSP_SEX`, `TRANSIT_DISTRICT`, `X_COORD_CD`, `Y_COORD_CD`, `Latitude`, `Longitude`, `Lat_Lon`, `New Georeferenced Column`, `LOC_OF_OCCUR_DES`.
- Rename columns that we are using, so they are easier to reference later on in our script.
    - BORO_NM -> borough
    - CMPLNT_FR_DT -> date
    - CMPLNT_FR_TM -> time
    - PREM_TYP_DESC -> premise
    - OFNS_DESC -> description
    - VIC_AGE_GROUP -> victim_age
    - VIC_SEX -> victim_sex
    - VIC_RACE -> victim_race
- The next preprocessing steps are all involving cleaning the columns that we are keeping, and they were processed as follows:
    - `borough`: Dropped records that contained nulls for borough.
    - `day_of_week`: We get the full name of the day of week the arrest occurred from the date column.
    - `month`: We get the month in groups of 2 from the date column, such that crimes that occurred in:
        - January or February mapped to 'JAN/FEB'
        - March or April mapped to 'MAR/APR'
        - May or June mapped to 'MAY/JUN'
        - July or August mapped to 'JUL/AUG'
        - September or October mapped to 'SEP/OCT'
        - November or December mapped to 'NOV/DEC'
    - `time_of_day`: We get the time of day by grouping the time column into more obvious periods of the day, such that in a 24 hour day:
        - 00:00 - 04:59 mapped to 'AFTER MIDNIGHT'
        - 5:00 - 08:59 mapped to 'MORNING RUSH HOUR'
        - 09:00 - 11:59 mapped to 'LATE MORNING'
        - 12:00 - 15:59 mapped to 'AFTERNOON'
        - 16:00 - 19:59 mapped to 'EVENING RUSH HOUR'
        - 20:00 - 23:59 mapped to 'NIGHT'
        We also remove records that do not contain a time.
    - `premise`: Remove rows that contain a non-descriptive premise such as 'MISCELLANEOUS', 'OTHER', or 'PHOTO/COPY'. Group together premise values that are similar to be equal in value, such as 'PARKING LOT/GARAGE (PUBLIC)' AND 'PARKING LOT/GARAGE (PRIVATE)' to be replaced with 'PARKING LOT'.
    - `description`: Remove rows that do not contain a description. Clean up descriptions that are non-descriptive such as 'OTHER STATE LAWS' and 'OTHER STATE LAWS (NON PENAL LA' and remove those rows. Concatenate descriptions that are similar to be equal, such as 'INTOXICATED/IMPAIRED DRIVING' and 'INTOXICATED & IMPAIRED DRIVING' to be replaced with 'IMPAIRED DRIVING'.
    - `victim_age`: Drop victim ages that contain the word 'UNKNOWN'.
    - `victim_sex`: Map 'F' to female and 'M' to male. The rest of the columns that either contain null or a different value will be dropped.
    - `victim_race`: Drop victim races that contain the word 'UNKNOWN'.
- After we get `day_of_week` and `month`, we remove the `date` column.
- After we get `time_of_day`, we remove the `time` column.
- Save the processed dataset to a CSV file named `INTEGRATED-DATASET.csv`

(c) Our choice of INTEGRATED-DATASET file is compelling as it will reveal statistics about crime throughout NYC. Through analyzing this dataset, we can figure out which boroughs have the highest crime rates, and whether certain boroughs have higher crime rates at certain periods of a year or days of a week, and even times of the day. Since we included days of the week that the arrest occurred, it allows for more visibility into whether certain areas are more dangerous over the weekends/weekdays. We've also cleaned up the descriptions to group together similar ones and drop the rows that provide no meaningful value to the dataset, which also helps in gauging the boroughs that have the highest serious crimes such as murder and kidnapping, compared to the less serious crimes such as petty theft. Due to constraints, we removed police precinct from the dataset, but it would provide some meaningful value in future iterations to run the algorithm based on the location of a police precinct, which will allow us to zone in on particular areas/districts that are close to where these arrests are taking place.

## Internal Design

The high-level process can be found in `main()` in `main.py`. We first read the min_support and the min_confidence arguments from the user (`get_arguments()`). Then, we load the integrated dataset (`read_csv()`), where each row is considered as a transaction. Before we can run the a-priori algorithm, we also extract the initial item sets (which are basically the individual items in each of the transaction) and their frequencies.  

We then feed the transactions, the initial item sets and their frequencies into the a-priori algorithm (`a_priori()`), which is based on the pseudocode in Sect. 2.1 of the Agrawal & Srikant (1994) paper. Before entering the loop where k increments, we calculate the support for each initial item set and only keep the ones with support >= min_support specified. This forms the set L(1) (and now k=2). Inside the loop, we first generate the candidates from L(k - 1) (`generate_candidates()`). The generation algorithm is logically the same as Sect. 2.1.1 of the paper. We apply a self-join on L(k - 1), which produces new item sets each with size k. Then, we prune the item sets which have any subset with size k - 1 not in L(k - 1). After the generation step, we check if the new item sets actually exist in the transactions, and if so we update their frequencies. The frequencies will be used in calculating supports and confidences later. In each iteration, if the support of a new valid item set is greater than or equal to min_support, we add it to the result container. We continue looping until L(k) becomes empty.  

Next, we use the eligible item sets from a-priori to build association rules (`build_association_rules()`). For each item set, we first find its subsets with any one item excluded. Thus, a subset will be the LHS of the association rule and the corresponding excluded item is the RHS of the rule. The rule confidence = support(LHS ∪ RHS) / support(LHS), where LHS ∪ RHS is just the item set being processed. As the number of transactions is cancelled out, confidence = frequency(item set) / frequency(LHS). If the confidence >= min_conf specified, we append the rule to the result container. Moreover, we filter some rules with an ignore set (details in Additional Information). Finally, the item sets and the rules with their supports/confidences are written to file (`write_output()`).

## Compelling Sample Run
`python3 main.py INTEGRATED-DATASET.csv 0.025 0.4`
This run provides 81 association rules, and contain several interesting results.

From the following association rules, we are able to see that victims of crimes that happen inside a residence in Brooklyn are most often Black females. The crimes that happen for females in Brooklyn are highly likely to occur inside a residence, which also suggests domestic violence crimes occur often in Brooklyn.
```
['FEMALE', 'RESIDENCE - PUBLIC HOUSING'] => ['BLACK'] (Conf: 60.2738%, Supp: 3.4511%)
['RESIDENCE - PUBLIC HOUSING'] => ['BLACK'] (Conf: 58.4099%, Supp: 4.6937%)
['BLACK', 'FEMALE', 'RESIDENCE - APT. HOUSE'] => ['BROOKLYN'] (Conf: 40.8671%, Supp: 3.0129%)
['ASSAULT', 'RESIDENCE - APT. HOUSE'] => ['BLACK'] (Conf: 41.7362%, Supp: 3.2886%)
['BROOKLYN', 'FEMALE', 'RESIDENCE - APT. HOUSE'] => ['BLACK'] (Conf: 50.4966%, Supp: 3.0129%)
['BLACK', 'RESIDENCE - APT. HOUSE'] => ['BROOKLYN'] (Conf: 40.1232%, Supp: 4.4594%)
['25-44', 'BROOKLYN', 'FEMALE'] => ['RESIDENCE - APT. HOUSE'] (Conf: 40.6523%, Supp: 3.1452%)
```

From the following association rules, we can gather that crimes that invole male victims usually occur on the street, rather than inside a resential area, building, or store. Further, the most common male victims are between the age of 25 and 44, and these type of crimes most commonly occur during the evening rush hour when people are commuting back home, night time, or after midnight between the hours of 12AM and 5AM. Further, the majority of the crimes for these victims include petty theft or theft of personal property. Additionally, it seems White Hispanics are the most common victims among these cases.
```
['GRAND LARCENY', 'MALE'] => ['STREET'] (Conf: 49.1939%, Supp: 4.2234%)
['MALE', 'PETIT LARCENY'] => ['STREET'] (Conf: 47.9386%, Supp: 4.1879%)
['25-44', 'GRAND LARCENY'] => ['STREET'] (Conf: 45.2995%, Supp: 3.4526%)
['25-44', 'MALE', 'WHITE HISPANIC'] => ['STREET'] (Conf: 45.1403%, Supp: 2.7915%)
['AFTER MIDNIGHT', 'MALE'] => ['STREET'] (Conf: 43.7842%, Supp: 3.2737%)
['MALE', 'WHITE HISPANIC'] => ['STREET'] (Conf: 43.6381%, Supp: 5.2007%)['EVENING RUSH HOUR', 'MALE'] => ['STREET'] (Conf: 43.403%, Supp: 4.6689%)
['BRONX', 'MALE'] => ['STREET'] (Conf: 43.0701%, Supp: 4.3642%)
['JUL/AUG', 'MALE'] => ['STREET'] (Conf: 42.9346%, Supp: 3.6618%)
['MALE', 'SEP/OCT'] => ['STREET'] (Conf: 42.7264%, Supp: 3.8732%)
['MALE', 'SUNDAY'] => ['STREET'] (Conf: 41.9831%, Supp: 2.7522%)
['MALE', 'MAY/JUN'] => ['STREET'] (Conf: 41.8718%, Supp: 3.4213%)
['25-44', 'MALE'] => ['STREET'] (Conf: 41.6423%, Supp: 9.6891%)
['MALE', 'NIGHT'] => ['STREET'] (Conf: 47.4524%, Supp: 4.4796%)
['MALE', 'QUEENS'] => ['STREET'] (Conf: 40.2798%, Supp: 4.6566%)

```

The following association rules provide additional insight about possible domestic violence cases. While most male crimes happen on the street, most female cases are inside residences. Victims between the ages of 25 and 44 further support this finding as the age range is common for mothers. These crimes occur with high support in Queens, with Bronx following closely after.
```
['25-44', 'BRONX', 'FEMALE'] => ['RESIDENCE - APT. HOUSE'] (Conf: 46.9595%, Supp: 2.9999%)
['BRONX', 'HARASSMENT'] => ['RESIDENCE - APT. HOUSE'] (Conf: 49.9573%, Supp: 2.9108%)
['25-44', 'RESIDENCE-HOUSE'] => ['QUEENS'] (Conf: 46.1916%, Supp: 2.7315%)
['RESIDENCE-HOUSE'] => ['QUEENS'] (Conf: 45.6666%, Supp: 6.0641%)
['FEMALE', 'RESIDENCE-HOUSE'] => ['QUEENS'] (Conf: 44.9971%, Supp: 3.5524%)
```

These association rules show that the majority of crimes involving a male of Asian or Pacific Islander descent is most likely to occur on the streets of Queens.
```
['ASIAN / PACIFIC ISLANDER', 'MALE'] => ['QUEENS'] (Conf: 44.9053%, Supp: 3.1292%)
['25-44', 'ASIAN / PACIFIC ISLANDER'] => ['QUEENS'] (Conf: 44.0291%, Supp: 2.5671%)
['ASIAN / PACIFIC ISLANDER', 'MALE'] => ['STREET'] (Conf: 40.5542%, Supp: 2.826%)
```

## Additional Information
A few considerations when creating our `INTEGRATED-DATASET` are as follows:
- Although demographics about the suspect may be interesting, we were concerned of racial profiling and the likes, so we decided not to include any information about the suspect of the crime, which includes their age, race, and sex. It's also likely that these were assumptions of the suspect's true identity, so it was reasonable to exclude it.
- From discussing with Professor Gravano (during OH and via email), he mentioned we could filter out the RHS to remove rules that provide no meaningful value and are trivial in nature. For this reason, we included an ignore set that filters out those unwanted RHS association rules inside `main.py` and filter out the association rules were the RHS is equal to the value in the ignore set. The professor mentioned that although this is a departure from the project description, it helps us filter out many rules that are not informative, which is inevitable to appear when running an association rule mining algorithm.
