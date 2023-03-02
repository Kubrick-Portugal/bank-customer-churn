
# %% --------------------------------------------------------------------------
# States
# -----------------------------------------------------------------------------
state_groups = {'California': ['California'], 
                'New York': ['New York'],
                'Texas': ['Texas'], 
                'Florida': ['Florida'], 
                'Illinois':['Illinois'],
                'Pennsylvania':['Pennsylvania'],
                'Ohio':['Ohio'],
                'Georgia':['Georgia'],
                'North Carolina':['North Carolina'],
                'Michigan':['Michigan'],
                'New Jersey':['New Jersey'],
                 'Virginia':['Virginia'],
                'Washington':['Washington'],
                'Arizona':['Arizona'],
                'Massachusetts':['Massachusetts'],
                'Tennessee':['Tennessee'],
                'Indiana':['Indiana'],
                'Missouri':['Missouri'],
                'Other':[
    'Alabama',
    'Alaska',
    'Arkansas',
    'Colorado',
    'Connecticut',
    'Delaware',
    'Hawaii',
    'Idaho',
    'Iowa',
    'Kansas',
    'Kentucky',
    'Louisiana',
    'Maine',
    'Maryland',
    'Minnesota',
    'Mississippi',
    'Montana',
    'Nebraska',
    'Nevada',
    'New Hampshire',
    'New Mexico',
    'North Dakota',
    'Oklahoma',
    'Oregon',
    'Rhode Island',
    'South Carolina',
    'South Dakota',
    'Utah',
    'Vermont',
    'West Virginia',
    'Wisconsin',
    'Wyoming'
]}
state_to_region = {}
for region, states in state_groups.items():
    for state in states:
        state_to_region[state] = region

# Apply the mapping to the 'state' column to create a new 'Region' column
df["region"] = df['state'].apply(lambda x: state_to_region[x] if x in state_to_region else 'Other')

# %% --------------------------------------------------------------------------
# Ages
# -----------------------------------------------------------------------------

age_groups = {
    '0-17': list(range(0, 18)),
    '18-25': list(range(18, 26)),
    '26-40': list(range(26, 41)),
    '41-64': list(range(41, 65)),
    '65+': list(range(65, 101))
}
age_to_agegroups = {}
for agegroups, age in age_groups.items():
    for ages in age:
        age_to_agegroups[ages] = agegroups

# Apply the mapping to the 'age' column to create a new 'agegroups' column
df["agegroups"] = df['age'].apply(lambda x: age_to_agegroups[x] if x in age_to_agegroups else None)
