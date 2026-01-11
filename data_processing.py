from dateutil import parser
import regex as re
import math
import json
import locale

# Parse 'value (label)' format
def parse_value_label(val):
    if isinstance(val, str):
        match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*\((.*?)\)\s*$", val)
        if match:
            return {
                "hours": float(match.group(1)) if '.' in match.group(1) else int(match.group(1)),
                "energy effect": match.group(2).strip()
            }
    return val  # return original if not matching format

def df_to_json(df):
    "Function to make dataframe with user data into JSON format"
    transformed = {}

    replacement_dict = {"Dagelijkse activiteit": "Dagelijkse activiteiten",
                   "Doel": "Doelen", 
                   "Symptoom": "Symptomen"}
    for date, fields in df.iterrows():
        entry = {}
        nested = {}

        for key, value in fields.items():
            if isinstance(value, float) and math.isnan(value):
                continue  # skip NaN values

        # Match anything like "Field name (category)"
            match = re.match(r"^(.*?)\s*\((.*?)\)\s*$", key)
            if match:
                variable = match.group(1).strip()
                category = match.group(2).strip().capitalize()
                if "Uur" in category:
                    category = category.replace("Uur) (", "")
                category = category.capitalize()
                for og, replace in replacement_dict.items():
                    if category==og:
                        category = replace
                parsed_value = parse_value_label(value)
                if category == "Doelen":
                    if int(parsed_value) == 1:
                        parsed_value = 'Behaald'
                    else: 
                        parsed_value = 'Niet behaald'

                if category not in nested:
                    nested[category] = []

                nested[category].append({
                    "variable": variable,
                    "value/score": parsed_value
                })
            else:
                entry[key.strip()] = value

    # Merge flat and nested fields
        entry.update(nested)
        transformed[date] = entry

# Output as formatted JSON
    json_output = json.dumps(transformed, indent=4, ensure_ascii=False)
    return json_output


def format_datum_column(df):
    # Try parsing each date flexibly
    def parse_date(d):
        return parser.parse(d, dayfirst=True)

    # Apply parsing
    df['Datum'] = df['Datum'].apply(parse_date)

    try:
        # Try using Dutch locale if available
        locale.setlocale(locale.LC_TIME, 'nl_NL.UTF-8')
        df['Datum'] = df['Datum'].dt.strftime('%A %d %B').str.lower()
    except locale.Error:
        # Fallback to manual translation if Dutch locale not available
        day_names = ['maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag', 'zondag']
        month_names = ['januari', 'februari', 'maart', 'april', 'mei', 'juni',
                       'juli', 'augustus', 'september', 'oktober', 'november', 'december']

        df['Datum'] = df['Datum'].apply(
            lambda d: f"{day_names[d.weekday()]} {d.day} {month_names[d.month - 1]}"
        )

    return df

def merge_activities(df):
    "Merges all columns containing (activiteit) in the header into one column."
        # here combine all activities columns into 1 column
    activity_cols = [col for col in df.columns if '(activiteit)' in col]

    df_long = df.melt(id_vars='Datum', value_vars=activity_cols,
                  var_name='Activiteit', value_name='Deelgenomen')

    # only get activities that were done
    df_filtered = df_long[df_long['Deelgenomen'] == 1]

    # remove the (activities) suffix
    # df_filtered['Activiteit'] = df_filtered['Activiteit'].str.replace(' \(activiteit\)', '', regex=True)
    df_filtered.loc[:, 'Activiteit'] = df_filtered['Activiteit'].str.replace(' \(activiteit\)', '', regex=True)

    df_grouped = df_filtered.groupby('Datum')['Activiteit'].apply(', '.join).reset_index()
    df_grouped.rename(columns={'Activiteit': 'Activiteiten'}, inplace=True)

    # Merge with original DataFrame
    df_merged = df.merge(df_grouped, on='Datum', how='left')

    # Drop the original activity columns
    df = df_merged.drop(columns=activity_cols)

    return df