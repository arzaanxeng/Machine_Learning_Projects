import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import preprocessor , helper
import matplotlib.pyplot as plt

df = pd.read_csv('Data/athlete_events.csv')
region_df = pd.read_csv('Data/noc_regions.csv')


df = preprocessor.preprocess(df , region_df)

st.markdown("""
    <div style="
        background: #047857;
        padding: 25px;
        border-radius: 18px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    ">
        <h2> Welcome to the Olympics Dashboard</h2>
        <p style="font-size:20px;">
            Discover Medal Tallies -- Athlete Achievements -- Trends across 28 Olympic editions
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/5/5c/Olympic_rings_without_rims.svg' width='300'>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("-Olympics Analysis-")
user_menu = st.sidebar.radio("Select an option ",("Medal Tally" , "Overall Analysis" , "Country Wise Analysis" , "Athlete Wise Analysis"))

if user_menu == 'Medal Tally':
    st.sidebar.header("Medal Tally")
    years,country = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox("Select Year",years)
    selected_country = st.sidebar.selectbox("Select Country", country)

    medal_tally = helper.fetch_medal_tally(df,selected_year,selected_country)
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Tally")
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " overall performance")
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(selected_country , " performance in " + str(selected_year) + " Olympics")
    st.table(medal_tally)

if user_menu == 'Overall Analysis':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.title("Top Statistics")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    # Plotting bar plot for participating nations over time
    nations_over_time = helper.participating_nations_overtime(df , 'region')
    st.title("Participating Nations over the Years")
    fig = px.bar(
        nations_over_time,
        x='Year',
        y='region',
        labels={'region': 'Number of Nations'},
        text='region'
    )
    fig.update_traces(marker_color='skyblue', textposition='outside')
    fig.update_layout(template='plotly_white', height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Plotting bar plot for participating athletes over time
    nations_over_time = helper.participating_nations_overtime(df , 'Name')
    st.title("Participating Athletes by Olympic Years")
    fig = px.bar(
        nations_over_time,
        x='Year',
        y='Name',
        labels={'Name': 'Number of Athletes'},
        text='Name'
    )
    fig.update_traces(marker_color='red', textposition='outside')
    fig.update_layout(template='plotly_white', height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Plotting bar plot for addition of sports over time
    nations_over_time = helper.participating_nations_overtime(df , 'Sport')
    st.title("Addition of different Sports over Olympic Years")
    fig = px.bar(
        nations_over_time,
        x='Year',
        y='Sport',
        labels={'Name': 'Number of Sports'},
        text='Sport'
    )
    fig.update_traces(marker_color='lightgreen', textposition='outside')
    fig.update_layout(template='plotly_white', height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Plotting a heatmap to analyze the addition of Sports over the years
    st.title("No. of Events over Time for Every Sport")
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    pivot = x.pivot_table(
        index='Sport',
        columns='Year',
        values='Event',
        aggfunc='count'
    ).fillna(0).astype('int')

    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(pivot,annot=True,fmt="d",cmap="Blues",linewidths=0.5,ax=ax
    )
    ax.set_title("Number of Events by Sport and Year")
    st.pyplot(fig)

    # Extracting most successful athletes

    st.title("Most Successful Athletes")
    sports_list = df['Sport'].unique().tolist()
    sports_list.sort()
    sports_list.insert(0, 'Overall')
    selected_sport = st.selectbox("Select Sport",sports_list )

    x = helper.most_successful_athletes(df,selected_sport)
    if selected_sport == 'Overall':
        st.subheader(f"Top Athletes - Overall")
    else:
        st.subheader(f"Top Athletes in {selected_sport}")

    st.dataframe(
        x.style.background_gradient(cmap='Blues', subset=['Total']),use_container_width=True,hide_index=True)


if user_menu == 'Country Wise Analysis':

    st.sidebar.title('Country Wise Analysis')
    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.sidebar.selectbox('Select a Country',country_list)
    country_df = helper.yearwise_medal_tally(df,selected_country)
    fig = px.line(country_df, x="Year", y="Medal")
    fig.update_layout(plot_bgcolor="white")
    st.title(selected_country + " Medal Tally over the years")
    st.plotly_chart(fig)


    st.title(selected_country + " excels in following sports indicated below")
    pt = helper.country_event_heatmap(df,selected_country)
    if not pt.empty:
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(pt, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No data available for selected filters")

    st.title("Top athletes of " + selected_country)
    top10_df = helper.most_successful_countrywise(df,selected_country)
    st.table(top10_df)


if user_menu == 'Athlete Wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],show_hist=False, show_rug=False)
    fig.update_layout(autosize=False,width=1000,height=600)
    st.title("Probability of winning a medal over the different ages ")
    st.plotly_chart(fig)

    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age wrt Sports")
    st.plotly_chart(fig)

    st.title("Men Vs Women Participation ")
    final = helper.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)






