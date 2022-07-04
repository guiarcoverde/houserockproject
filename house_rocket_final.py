import pandas as pd
import streamlit as st
import numpy as np
import folium
import geopandas
import plotly.express as px

from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from datetime import datetime


st.set_page_config(layout='wide')

@st.cache(allow_output_mutation=True)

def data(path):
    kchouse = pd.read_csv(path)

    return kchouse

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file( url )
    
    return geofile

def set_feature(kchouse):
    sqft_to_m2 = kchouse['sqft_lot'] / 10.764
    kchouse['price_m2'] = kchouse['price'] / sqft_to_m2

    return kchouse

def ds_overview(kchouse):
    try:
        
        show_attributes = st.sidebar.multiselect('Please enter the desired columns', kchouse.columns)
        zip_filter = st.sidebar.multiselect('Please enter the Zipcode:', kchouse['zipcode'].unique())
        kchouse['date'] = pd.to_datetime(kchouse['date']).dt.strftime('%Y-%m-%d')
        st.title('Data Overview')
    
        if (show_attributes != []) & (zip_filter != []):
            kchouse = kchouse.loc[kchouse['zipcode'].isin(zip_filter), show_attributes]

        elif (show_attributes == []) & (zip_filter != []):
            kchouse = kchouse.loc[kchouse['zipcode'].isin(zip_filter), :]

        elif (show_attributes != []) & (zip_filter == []):
            kchouse = kchouse.loc[:, show_attributes]

        else:
            kchouse = kchouse.copy()

        st.dataframe(kchouse.head(500).style.format(formatter={'lat':'{:.4f}', 'long':'{:.3f}'},precision=2))
    
    
        c1, c2 = st.columns((1, 1))
    
        houses_per_zip = kchouse[['id', 'zipcode']].groupby('zipcode').count().reset_index()
        prices_mean_per_zip = kchouse[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
        sqft_per_zip_mean = kchouse[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
        m2_price_mean = kchouse[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()
    
        merges1 = pd.merge(houses_per_zip, prices_mean_per_zip, on='zipcode', how='inner')
        merge2 = pd.merge(merges1, sqft_per_zip_mean, on='zipcode', how='inner')
        kchouse_1 = pd.merge(merge2, m2_price_mean, on='zipcode', how='inner')

        kchouse_1.columns = ['Zipcode', 'Total houses', 'Price', 'Living Room Size', 'Price per M²']
        c1.header('Average Values')
        c1.dataframe(kchouse_1.style.format(precision=2), height = 600)
    
        num_attributes = kchouse.select_dtypes(include=['int64', 'float64'])
        means = pd.DataFrame(num_attributes.apply(np.mean))
        medians = pd.DataFrame(num_attributes.apply(np.median))
        stdeviation = pd.DataFrame(num_attributes.apply(np.std))
        maximums = pd.DataFrame(num_attributes.apply(np.max))
        minimuns = pd.DataFrame(num_attributes.apply(np.min))   
    
        kchouse_2 = pd.concat([maximums, minimuns, means, medians, stdeviation], axis= 1).reset_index()
        kchouse_2.columns = ['attributes', 'Max Value', 'Minimum Value', 'Means', 'Medians', 'Standard Deviation']
        c2.header('Descriptive Statistics')
        c2.dataframe(kchouse_2.style.format(precision=2), height = 600)

        

    except KeyError:
        st.subheader('Please select the attributes: id, zipcode, price, sqft_living and price_m2 for average values and descriptive analysis')
    
    return None

def density_maps(kchouse, geofile):
    st.title('Region Overview')

    map1, map2 = st.columns((1, 1))
    map1.header('Region Density')
    df = kchouse.sample(500)
    density_map = folium.Map(location= [kchouse['lat'].mean(), kchouse['long'].mean()], default_zoom_start=15)

    makecluster = MarkerCluster().add_to(density_map)
    kchouse['date'] = pd.to_datetime(kchouse['date']).dt.strftime('%Y-%m-%d')
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']], popup='Sold for U${0} on {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, yr built: {5}'.format(row['price'], row['date'], row['sqft_living'], row['bedrooms'], row['bathrooms'], row['yr_built'])).add_to(makecluster)

    with map1:
        folium_static(density_map)


    df_1 = kchouse.sample(500)
    df_1 = kchouse[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df_1.columns = ['Zip', 'Price']
    geofile = geofile[geofile['ZIP'].isin(df_1['Zip'].tolist())]
    map2.header('Price Density')
    price_map = folium.Map(location= [kchouse['lat'].mean(), kchouse['long'].mean()], default_zoom_start=15)
    price_map.choropleth(data= df_1, geo_data=geofile, columns=['Zip', 'Price'], key_on='feature.properties.ZIP', fill_color='YlOrRd', fill_opacity=0.7, fill_line=0.2, legend_name='Avg Price' )

    with map2:
        folium_static(price_map)
    
    return None

def commercial_attributes(kchouse):
    st.sidebar.header('Commercial Options')
    #Graph with year filter
    st.sidebar.subheader('Please Select the year built')
    yr_built_filter = st.sidebar.slider('Year Built', int(kchouse['yr_built'].min()), int(kchouse['yr_built'].max()), int(kchouse['yr_built'].min()) )
    st.header('Average prices per year')
    price_by_year = kchouse.loc[kchouse['yr_built'] <= yr_built_filter]
    price_by_year = price_by_year[['price', 'yr_built']].groupby('yr_built').mean().reset_index()
    graph_by_year = px.line(price_by_year, x='yr_built', y='price')
    st.plotly_chart(graph_by_year, use_container_width=True)
    
    #Graph with day filter
    kchouse['date'] = pd.to_datetime(kchouse['date']).dt.strftime('%Y-%m-%d')

    st.sidebar.subheader('Please select the day desired')
    day_filter = st.sidebar.slider('Day Selection', datetime.strptime(kchouse['date'].min(), '%Y-%m-%d'), datetime.strptime(kchouse['date'].max(), '%Y-%m-%d'), datetime.strptime(kchouse['date'].min(), '%Y-%m-%d') )
    kchouse['date'] = pd.to_datetime(kchouse['date'])
    st.header('Average Prices per Day')
    price_by_day = kchouse.loc[kchouse['date'] < day_filter]
    price_by_day = price_by_day[['price', 'date']].groupby('date').mean().reset_index()
    graph_by_day = px.line(price_by_day, x='date', y= 'price')
    st.plotly_chart(graph_by_day, use_container_width=True)
    
    #Price Histogram
    st.sidebar.header('Price Distribution')
    st.header('Price distribution')
    st.sidebar.subheader('Select the price limit')

    price_filter = st.sidebar.slider('Price', int(kchouse['price'].min()), int(kchouse['price'].max()), int(kchouse['price'].mean()))
    price_limit = kchouse.loc[kchouse['price'] <= price_filter]
    price_dist = px.histogram(price_limit, x='price', nbins=50)
    st.plotly_chart(price_dist, use_container_width=True)
    
    return None

def attributes_dist(kchouse):
    # Houses separated by attributes
    st.header('Houses per attributes')
    st.sidebar.header('Attributes')
    atr1, atr2 = st.columns(2)
    
    #Houses per bedrooms
    atr1.subheader('Houses per bedrooms')
    bedrooms_filter = st.sidebar.selectbox('Please select the number of maximum bedrooms', kchouse['bedrooms'].sort_values().unique())
    bedrooms_limit = kchouse.loc[kchouse['bedrooms'] <= bedrooms_filter]
    bedrooms_dist = px.histogram(bedrooms_limit,x = 'bedrooms', nbins=19)
    atr1.plotly_chart(bedrooms_dist, use_container_width=True)
    
    #Houses per bathrooms
    bathrooms_filter = st.sidebar.selectbox('Please select the number of maximum bathrooms', kchouse['bathrooms'].sort_values().unique())
    atr2.subheader('Houses per bathrooms')
    bathrooms_limit = kchouse.loc[kchouse['bathrooms'] <= bathrooms_filter]
    bathrooms_dist = px.histogram(bathrooms_limit,x = 'bathrooms', nbins=19)
    atr2.plotly_chart(bathrooms_dist, use_container_width=True)

    #Houses per floors
    st.subheader('Houses per floor number')
    floors_filter = st.sidebar.selectbox('Please select the number of maximum floors', kchouse['floors'].sort_values().unique())
    floors_limit = kchouse.loc[kchouse['floors'] <= floors_filter]
    floors_dist = px.histogram(floors_limit,x = 'floors', nbins=19)
    st.plotly_chart(floors_dist, use_container_width=True)

    #Houses with waterfront view
    waterview_filter = st.sidebar.checkbox('Waterfront View')

    if waterview_filter:
        waterview_limit = kchouse[kchouse['waterfront'] == 1]
        waterview_dist = px.histogram(waterview_limit, x='waterfront', nbins=10)
        st.plotly_chart(waterview_dist, use_container_width=True)

    else:
        no_waterview = kchouse[kchouse['waterfront'] == 0]
        no_waterview_dist = px.histogram(no_waterview, x='waterfront', nbins=10)
        st.plotly_chart(no_waterview_dist, use_container_width=True)

    
    

    
    return None

if __name__ == '__main__':
    
    url = 'Zip_Codes.geojson'
    path = 'kc_house_data.csv'
    
    geofile = get_geofile( url )
    kchouse = data(path)
    
    kchouse = set_feature(kchouse)

    ds_overview(kchouse)

    density_maps(kchouse, geofile)

    commercial_attributes(kchouse)

    attributes_dist(kchouse)