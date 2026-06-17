import pandas as pd
def medal_tally(df):
    medal_tally = df.drop_duplicates(subset=['Team', 'region', 'NOC', 'Games', 'Season', 'City', 'Sport', 'Event', 'Medal'])
    medal_tally = medal_tally.groupby('NOC').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending=False).reset_index()
    medal_tally['total'] = medal_tally['Gold'] + medal_tally['Silver'] + medal_tally["Bronze"]
    return medal_tally

def country_year_list(df):
    years = sorted(df["Year"].unique())
    years.insert(0, 'Overall')

    country = sorted(df['region'].dropna().unique())
    country.insert(0, 'Overall')

    return years, country


def fetch_medal_tally(df, year, country):
    medal_df = df.drop_duplicates(
        subset=['Team', 'region', 'NOC', 'Games', 'Season', 'City', 'Sport', 'Event', 'Medal'])
    flag = 0
    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df

    elif year == 'Overall' and country != 'Overall':
        temp_df = medal_df[medal_df['region'] == country]
        flag = 1

    elif year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]

    else:
        temp_df = medal_df[
            (medal_df['Year'] == int(year)) &
            (medal_df['region'] == country)]

    if flag == 1:
        x = temp_df.groupby('Year')[['Gold', 'Silver', 'Bronze']].sum()
        x = x.sort_values('Year').reset_index()

    else:
        x = temp_df.groupby('NOC')[['Gold', 'Silver', 'Bronze']].sum()
        x = x.sort_values('Gold', ascending=False).reset_index()

    x['Total'] = x['Gold'] + x['Silver'] + x['Bronze']

    return x

def participating_nations_overtime(df , col ):
    analysis = df.drop_duplicates(['Year', col])['Year'].value_counts().reset_index().sort_values('Year')
    analysis = analysis.rename(columns={'count': col})
    return analysis

def most_successful_athletes(df, sport):
    temp_df = df.dropna(subset=['Medal'])
    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]
    x = temp_df['Name'].value_counts().reset_index()
    x.columns = ['Name', 'Total']
    x = x.head(12).merge(
        df[['Name', 'Sport', 'region']].drop_duplicates('Name'),
        on='Name',
        how='left'
    )
    return x

def yearwise_medal_tally(df , country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    final_df = new_df.groupby('Year').count()['Medal'].reset_index()
    return final_df

def country_event_heatmap(df,country):
    x = df.dropna(subset=['Medal'])
    x = x.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    new_df = x[x['region'] == country]

    plot = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)
    return plot



def most_successful_countrywise(df, country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df = temp_df[temp_df['region'] == country]
    x = (temp_df['Name'].value_counts().reset_index(name='Medals').head(10))
    x = x.merge(temp_df[['Name', 'Sport']], on='Name',how='left').drop_duplicates('Name').reset_index(drop=True)
    return x

def men_vs_women(df):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()
    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)
    final.fillna(0, inplace=True)
    return final