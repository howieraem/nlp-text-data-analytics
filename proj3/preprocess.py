import pandas as pd
from datetime import datetime


# Helper functions to convert date
def get_weekday(date_to_convert):
    return datetime.strptime(date_to_convert, '%m/%d/%Y').strftime('%A').upper()


# def get_month(date_to_convert):
#     return datetime.strptime(date_to_convert, '%m/%d/%Y').strftime('%B').upper()


def get_month(date_to_convert):
    month = int(datetime.strptime(date_to_convert, '%m/%d/%Y').strftime('%m'))
    if month <= 2:
        return "JAN/FEB"
    elif month <= 4:
        return "MAR/APR"
    elif month <= 6:
        return "MAY/JUN"
    elif month <= 8:
        return "JUL/AUG"
    elif month <= 10:
        return "SEP/OCT"
    else:
        return "NOV/DEC"


def get_time_of_day(time):
    hour = datetime.strptime(time, '%H:%M:%S').hour
    if hour < 5 or hour >= 24:
        return "AFTER MIDNIGHT"
    elif hour >= 5 and hour < 9:
        return "MORNING RUSH HOUR"
    elif hour < 12:
        return "LATE MORNING"
    elif hour < 16:
        return "AFTERNOON"
    elif hour < 20:
        return "EVENING RUSH HOUR"
    elif hour < 24:
        return "NIGHT"


df = pd.read_csv('NYPD_Complaint_Data_Current__Year_To_Date_.csv')

# Remove unused columns
unused_cols = (
    'CMPLNT_NUM',
    'ADDR_PCT_CD',
    'CMPLNT_TO_DT',
    'CMPLNT_TO_TM',
    'CRM_ATPT_CPTD_CD',
    'HADEVELOPT',
    'HOUSING_PSA',
    'JURISDICTION_CODE',
    'JURIS_DESC',
    'KY_CD',
    'LAW_CAT_CD',
    'PARKS_NM',
    'PATROL_BORO',
    'PD_CD',
    'PD_DESC',
    'RPT_DT',
    'STATION_NAME',
    'SUSP_AGE_GROUP',
    'SUSP_RACE',
    'SUSP_SEX',
    'TRANSIT_DISTRICT',
    'X_COORD_CD',
    'Y_COORD_CD',
    'Latitude',
    'Longitude',
    'Lat_Lon',
    'New Georeferenced Column',
    'LOC_OF_OCCUR_DESC'
    # 'VIC_SEX',
    # 'VIC_AGE_GROUP'
)

for col in unused_cols:
    df.drop(col, axis=1, inplace=True)

# Rename columns
df.rename(
    columns={
        'BORO_NM': 'borough',
        'CMPLNT_FR_DT': 'date',
        'CMPLNT_FR_TM': 'time',
        # 'LOC_OF_OCCUR_DESC': 'location',
        'PREM_TYP_DESC': 'premise',
        'OFNS_DESC': 'description',
        'VIC_AGE_GROUP': 'victim_age',
        'VIC_SEX': 'victim_sex',
        'VIC_RACE': 'victim_race',
    },
    inplace=True
)

# Replace victim_sex with full name
df['victim_sex'] = df['victim_sex'].map({
    'F': 'FEMALE',
    'M': 'MALE',
})

# Merge age groups
# df['victim_age'] = df['victim_age'].map({
#     '<18': '<25',
#     '18-24': '<25',
#     '25-44': '25-44',
#     '45-64': '>44',
#     '65+': '>44',
# })
df.drop(df.loc[df['victim_age'] == 'UNKNOWN'].index, inplace=True)
df.drop(df.loc[df['victim_race'] == 'UNKNOWN'].index, inplace=True)

# Remove rows that contain null values
df.dropna(inplace=True)

# Clean up descriptions
df['description'].replace({
    'ASSAULT 3 & RELATED OFFENSES': 'ASSAULT',
    'FELONY ASSAULT': 'ASSAULT',
    'MISCELLANEOUS PENAL LAW': 'MISCELLANEOUS',
    'NYS LAWS-UNCLASSIFIED FELONY': 'MISCELLANEOUS',
    'GRAND LARCENY OF MOTOR VEHICLE': 'GRAND LARCENY',
    'OTHER OFFENSES RELATED TO THEF': 'THEFT',
    'PROSTITUTION & RELATED OFFENSES': 'SEX CRIMES',
    'CRIMINAL MISCHIEF & RELATED OF': 'MISCELLANEOUS',
    'MURDER & NON-NEGL. MANSLAUGHTE': 'MURDER',
    'BURGLARY': 'THEFT',
    'OFF. AGNST PUB ORD SENSBLTY &': 'MISCELLANEOUS',
    'OTHER STATE LAWS (NON PENAL LA': 'MISCELLANEOUS',
    'FRAUDS': 'FRAUD',
    'OFFENSES INVOLVING FRAUD': 'FRAUD',
    'OTHER STATE LAWS': 'MISCELLANEOUS',
    'THEFT-FRAUD': 'FRAUD',
    'INTOXICATED/IMPAIRED DRIVING': 'IMPAIRED DRIVING',
    'INTOXICATED & IMPAIRED DRIVING': 'IMPAIRED DRIVING',
    'ADMINISTRATIVE CODE': 'MISCELLANEOUS',
    'BURGLAR\'S TOOLS': 'THEFT',
    'KIDNAPPING & RELATED OFFENSES': 'KIDNAPPING',
    'HOMICIDE-NEGLIGENT,UNCLASSIFIE': 'MURDER',
    'AGRICULTURE & MRKTS LAW-UNCLASSIFIED': 'MISCELLANEOUS',
    'HARRASSMENT 2': 'HARASSMENT',
    'OTHER STATE LAWS (NON PENAL LAW)': 'MISCELLANEOUS',
    'LOITERING/GAMBLING (CARDS, DIC': 'GAMBLING',
    'KIDNAPPING AND RELATED OFFENSES': 'KIDNAPPING',
    'ADMINISTRATIVE CODES': 'MISCELLANEOUS',
    'FELONY SEX CRIMES': 'SEX CRIMES',
    'NEW YORK CITY HEALTH CODE': 'MISCELLANEOUS'
}, inplace=True)

df.drop(df.loc[df['description'] == 'MISCELLANEOUS'].index, inplace=True)


# Clean up premise
df['premise'].replace({
    # 'RESIDENCE - APT. HOUSE': 'RESIDENCE',
    'PARKING LOT/GARAGE (PUBLIC)': 'PARKING LOT',
    'PARKING LOT/GARAGE (PRIVATE)': 'PARKING LOT',
    # 'RESIDENCE-HOUSE': 'RESIDENCE',
    # 'RESIDENCE - PUBLIC HOUSING': 'RESIDENCE',
    # 'MAILBOX INSIDE': 'RESIDENCE',
    # 'MAILBOX OUTSIDE': 'RESIDENCE',
    'STORE UNCLASSIFIED': 'STORE/RESTAURANT',
    'GROCERY/BODEGA': 'STORE/RESTAURANT',
    'FOOD SUPERMARKET': 'STORE/RESTAURANT',
    'DRUG STORE': 'STORE/RESTAURANT',
    'CHAIN STORE': 'STORE/RESTAURANT',
    'CLOTHING/BOUTIQUE': 'STORE/RESTAURANT',
    'DEPARTMENT STORE': 'STORE/RESTAURANT',
    'TELECOMM. STORE': 'STORE/RESTAURANT',
    'VARIETY STORE': 'STORE/RESTAURANT',
    'RESTAURANT/DINER': 'STORE/RESTAURANT',
    'FAST FOOD': 'STORE/RESTAURANT',
    'TAXI (LIVERY LICENSED)': 'TAXI',
    'TAXI/LIVERY (UNLICENSED)': 'TAXI',
    'TAXI (YELLOW LICENSED)': 'TAXI'
}, inplace=True)
df.drop(df.loc[df['premise'] == 'MISCELLANEOUS'].index, inplace=True)
df.drop(df.loc[df['premise'] == 'OTHER'].index, inplace=True)
df.drop(df.loc[df['premise'] == 'BOOK/CARD'].index, inplace=True)
df.drop(df.loc[df['premise'] == 'SOCIAL CLUB/POLICY'].index, inplace=True)
df.drop(df.loc[df['premise'] == 'PHOTO/COPY'].index, inplace=True)


# Get day of week from the date
df['day_of_week'] = df['date'].apply(get_weekday)

# Get month from the date
df['month'] = df['date'].apply(get_month)

# Get time of day from the time
df['time_of_day'] = df['time'].apply(get_time_of_day)

# Drop date and time columns
df.drop('date', axis=1, inplace=True)
df.drop('time', axis=1, inplace=True)

# Save
df.to_csv('INTEGRATED-DATASET.csv', index=False, header=None)
