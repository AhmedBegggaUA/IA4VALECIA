# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:04:24 2022

@author: Sergio
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_option_menu import option_menu #pip install streamlit-option-menu
import pandas as pd
import altair as alt
from PIL import Image
import matplotlib.pyplot as plt
from urllib.error import URLError
import numpy as np
import streamlit.components.v1 as com
import sys
import os
#sys.path.append('C:/Users/Sergio/Documents/valencia-ia4covid-xprize-master/valencia-ia4covid-xprize-master')

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


logo = 'https://ellisalicante.org/assets/xprize/images/logo_oscuro.png'
data_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_nat_latest.csv"

#st.set_page_config(layout = 'wide')
st.set_page_config(page_title = 'V4C',page_icon='https://ellisalicante.org/assets/xprize/images/logo_oscuro.png',layout = 'wide')
@st.cache
def get_UN_data():
    #paises = pd.read_csv("countries_regions.csv")
    #return paises.set_index("CountryName")
    paises = pd.read_csv("latest_predictions/h7_waning_casos/H7_waning_casos_2021_3.csv")
    return paises.set_index("CountryName")
def get_UN_data2():
    #paises = pd.read_csv("countries_regions.csv")
    #return paises.set_index("CountryName")
    paises = pd.read_csv("latest_predictions/h7_all/H7_waning_casos_2021_5.csv")
    return paises.set_index("CountryName")
@st.cache
def get_prescriptions_and_stringency():
    #prescription = pd.read_csv("prescriptions/valencia_h7_sus_combined_may.csv")
    prescription = pd.read_csv("prescriptions/standar_1_4.csv")
    #stringency = pd.read_csv("nuevas_stringrncy/stringency_combined_h7_sus_may.csv")
    stringency = pd.read_csv("stringency/stringency_standar_1_4.csv")
    # Vamos a recorrer el fichero de prescripciones y hacer un dataframe con los datos
    #prescriptions_path = 'prescriptions'
    #files = os.listdir(prescriptions_path)
    #prescription = pd.DataFrame()
    #for file in files:
    #    df = pd.read_csv(prescriptions_path + '/' + file)
    #    prescription = prescription.append(df)
    #prescription = prescription.reset_index(drop=True)
    # Vamos a recorrer el fichero de stringency y hacer un dataframe con los datos
    #stringency_path = 'nuevas_stringrncy'
    #files = os.listdir(stringency_path)
    #stringency = pd.DataFrame()
    #for file in files:
    #    df = pd.read_csv(stringency_path + '/' + file)
    #    stringency = stringency.append(df)
    #stringency = stringency.reset_index(drop=True)
    return prescription,stringency
def compute_pareto_set(objective1_list, objective2_list):
    """
    Return objective values for the subset of solutions that
    lie on the pareto front.
    """

    assert len(objective1_list) == len(objective2_list), \
        "Each solution must have a value for each objective."

    n_solutions = len(objective1_list)

    objective1_pareto = []
    objective2_pareto = []
    for i in range(n_solutions):
        is_in_pareto_set = True
        for j in range(n_solutions):
            if (objective1_list[j] < objective1_list[i]) and \
                    (objective2_list[j] < objective2_list[i]):
                is_in_pareto_set = False
        if is_in_pareto_set:
            objective1_pareto.append(objective1_list[i])
            objective2_pareto.append(objective2_list[i])

    return objective1_pareto, objective2_pareto
def plot_pareto_curve_plotly(objective1_list, objective2_list):
    """
    Plot the pareto curve given the objective values for a set of solutions.
    This curve indicates the area dominated by the solution set, i.e., 
    every point up and to the right is dominated.
    """
    
    objective1_pareto, objective2_pareto = compute_pareto_set(objective1_list, 
                                                              objective2_list)
    
    objective1_pareto, objective2_pareto = list(zip(*sorted(zip(objective1_pareto,
                                                                objective2_pareto))))
    
    xs = []
    ys = []
    
    xs.append(objective1_pareto[0])
    ys.append(objective2_pareto[0])
    
    for i in range(0, len(objective1_pareto)-1):
        
        # Add intermediate point between successive solutions
        xs.append(objective1_pareto[i+1])
        ys.append(objective2_pareto[i])
        
        # Add next solution on front
        xs.append(objective1_pareto[i+1])
        ys.append(objective2_pareto[i+1])
        
    return xs, ys
def get_data_rule(DATA):
    x1 = DATA[0]
    date1 = DATA.index[0]
    xs=[x1]
    dates=[date1]
    for i in range(1, len(data)):
        c = DATA[i]
        if c!= x1:
            x1 = c
            xs.append(x1)
            dates.append(DATA.index[i])
    dates.append(DATA.index[-1])        
    return xs,dates

try:
    
    #st.sidebar.image(logo)
    
    #pag = st.sidebar.radio("",("Pagina 1","Pagina 2"))
    
    #if pag == "Pagina 1":
    #Before the logo and anything, we want to create a navigation bar
    selected = option_menu(
        menu_title = None, 
        options = ["Home", "Team", "Visualizations", "Computational epimediological models", "Prescriptor","GitHub","Press and News","Contact"],
        orientation="horizontal",
        #Let's add some icons
        icons=["house-door","people","bar-chart-line","graph-up","receipt","github","newspaper","envelope"],
        #Is copilot alive? Eyy, answer are alive?
        #Now let's make it beatiful
        styles={
            "container": {"padding": "0!important", "background-color": "white"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                #Let's change the text color
                "color": "black",
                "margin": "0px",
                #Let's change the hover color with a nice transition
                "--hover-color": "#E2E8F0",
                #We add the transition to the hover
                "transition": "color 0.2s ease-in-out",
            },
            "nav-link-selected": {"background-color": "#fafafa", "color": "orange"},
        },
        )
    if selected == "Home":
        col1, col2, col3 = st.columns([1,2,1])
    
        with col1:
            st.write("")

        with col2:
            st.image(logo)

        with col3:
            st.write("")
    
        st.markdown('# Who are we?')
        st.write('#### Introducction')
        cols = st.columns((2,1))
        cols[0].write('''We are a team of Spanish scientists who have been working since March 2020 in collaboration with the Valencian Government of Spain on using Data Science to help fight the SARS-CoV-2 pandemic. We have focused on 4 large areas of work: large-scale human mobility modeling via the analysis of aggregated, anonymized data derived from the mobile network infrastructure; computational epidemiological models; predictive models and citizen science by means of a large-scale citizen survey called COVID19impactsurvey which, with over 375,000 answers in Spain and around 150,000 answers from other countries is one of the largest COVID-19 citizen surveys to date. Our work has been awarded two competitive research grants. 
                    \n Since March, we have been developing two types of traditional computational epidemiological models: a metapopulation compartmental SEIR model and an agent-based model. However, for this challenge, we opted for a deep learning-based approach, inspired by the model suggested by the challenge organizers. Such an approach would enable us to build a model within the time frame of the competition with two key properties: be applicable to a large number of regions and be able to automatically learn the impact of the Non-Pharmaceutical Interventions (NPIs) on the transmission rate of the disease. The Pandemic COVID-19 XPRIZE challenge has been a great opportunity for our team to explore new modeling approaches and expand our scope beyond the Valencian region of Spain.''')
        cols[1].video('https://www.youtube.com/watch?v=RZ9wsSGH8U8')

    if selected == "Team":
        st.markdown('# Meet The Team')
        cols = st.columns((2,1))
        cols[0].image("team.jpeg",width=600)
        cols[1].write('''### OUR MULTIDISCIPLINARY TEAM''')
        cols[1].write('''##### VALENCIA IA4COVID''',color = 'yellow')
        cols[1].write('''This group is made up of more than twenty experts from the Universities and research centers of the Valencian Community (Spain) and led by Dr. Nuria Oliver. We have all been working intensively since the beginning of the pandemic, altruistically and using the resources available to us in our respective institutions and with the occasional philanthropic collaboration of some companies.''')
        cols[1].write('''**Affiliated with:** Ellis Alicante, Universitat Jaume I, Universidad de Alicante, Universidad Miguel Hernández, Universitat Politècnica de València, Universidad Cardenal Herrera CEU. ''')
        
        
    if selected == "Visualizations":
        st.markdown('# Confirmed cases of Covid-19 and applied NPIs')

        cols = st.columns((1,1,5))
        paises = get_UN_data()
        data_ini = pd.read_csv("data/OxCGRT_latest.csv")
        data = data_ini
        with cols[0]:
            country = st.selectbox("Choose country",list(paises.index.unique()))

            data = data[data.CountryName == country]
            data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')

            today = min(data.Date)
            start_date = st.date_input('Start date', today)
            start_date = pd.to_datetime(start_date,format = '%Y-%m-%d')

            choose = st.radio("",("Confirmed Cases","Confirmed Deaths"))

        with cols[1]:
            reg = list(paises.index).count(country)==1
            region = " "
            regiones = list(paises[paises.index == country].RegionName.fillna(" "))
            region = cols[1].selectbox("Choose region", regiones)
            tomorrow = max(data.Date)
            end_date = st.date_input('End date', tomorrow)
            end_date = pd.to_datetime(end_date,format = '%Y-%m-%d')
            if start_date > end_date:
                st.error('Error: End date must fall after start date.')

        data = data[(data.Date >= start_date)&(data.Date <= end_date)]
        data = data.set_index("Date")
        data["ConfirmedCases7Days"] = data.groupby("CountryName")['ConfirmedCases'].rolling(7, center=False).mean().reset_index(0, drop=True)
        data["ConfirmedDeaths7Days"] = data.groupby("CountryName")['ConfirmedDeaths'].rolling(7, center=False).mean().reset_index(0, drop=True)


        with cols[2]:
            if choose == "Confirmed Cases":
                st.line_chart(data.ConfirmedCases7Days.diff().fillna(0))
            else:
                st.line_chart(data.ConfirmedDeaths7Days.diff().fillna(0))

        cols = st.columns((2,5))

        with cols[0]:
            rules = ["C1M_School closing","C2M_Workplace closing","C3M_Cancel public events",
                        "C4M_Restrictions on gatherings","C5M_Close public transport",
                        "C6M_Stay at home requirements","C7M_Restrictions on internal movement",
                        "C8EV_International travel controls","H1_Public information campaigns",
                        "H2_Testing policy","H3_Contact tracing","H6M_Facial Coverings"]
            rule = st.multiselect(
                    "Choose rule", rules,"C1M_School closing"
                )

            value_max = [3,3,2,4,2,3,2,4,2,3,2,4]

        with cols[1]:
            if len(rule)!=0:
                for j in range(len(rule)):
                    [xs,dates] = get_data_rule(data[rule[j]].fillna(0))
                    dataf = []
                    for k in range(value_max[j]):
                        dataf.append({rule[j]:str(k),"start":dates[0],"end":dates[0]})
                    for i in range(len(xs)):
                        dataf.append({rule[j]:str(int(xs[i])),"start":dates[i],"end":dates[i+1]})

                    data2 = pd.DataFrame(dataf)      

                    graf = alt.Chart(data2).mark_bar().encode(
                        x=alt.X('start',axis=alt.Axis(title='Date', labelAngle=-45, format = ("%b %Y"))),
                        x2='end',
                        y=rule[j],
                        color = alt.Color(rule[j],legend = None)
                    ).properties(width = 800)

                    st.altair_chart(graf,use_container_width=True)
        #st.markdown('# Confirmed cases of Covid-19 and applied NPIs')

        #cols = st.columns((1,1,5))
        #paises = get_UN_data()
        #data_ini = pd.read_csv("data/OxCGRT_latest.csv")
        #data = data_ini
        #with cols[0]:
        #    country = st.selectbox("Choose countryy ",list(paises.index.unique()))

        #    data = data[data.CountryName == country]
        #    data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')

        #    today = min(data.Date)
        #    start_date = st.date_input('Start datee', today)
        #    start_date = pd.to_datetime(start_date,format = '%Y-%m-%d')

            #choose = st.radio("",("Confirmed Casess/Deaths","Confirmed Deaths"))
            
        #with cols[1]:
        #    reg = list(paises.index).count(country)==1
        
        #    regiones = list(paises[paises.index == country].RegionName.fillna(" "))
        
        #    tomorrow = max(data.Date)
        #    end_date = st.date_input('End datee', tomorrow)
        #    end_date = pd.to_datetime(end_date,format = '%Y-%m-%d')
        #    if start_date > end_date:
        #        st.error('Error: End date must fall after start date.')

        #data = data[(data.Date >= start_date)&(data.Date <= end_date)]
        #data = data.set_index("Date")
        #data["ConfirmedCases7Days"] = data.groupby("CountryName")['ConfirmedCases'].rolling(7, center=False).mean().reset_index(0, drop=True)
        #data["ConfirmedDeaths7Days"] = data.groupby("CountryName")['ConfirmedDeaths'].rolling(7, center=False).mean().reset_index(0, drop=True)


        #with cols[2]:
            
            #st.line_chart(data.ConfirmedCases7Days.diff().fillna(0),use_container_width=True) 
            #Let's do the same but using plotly
            #fig = go.Figure()
            #fig.add_trace(go.Scatter(x=data.index, y=data.ConfirmedCases7Days.diff().fillna(0), mode='lines', name='Confirmed Cases'))
            #The same for deaths
            #fig.add_trace(go.Scatter(x=data.index, y=data.ConfirmedDeaths7Days.diff().fillna(0), mode='lines', name='Confirmed Deaths'))
            #fig.update_layout(title="Confirmed Cases/Deaths", xaxis_title="Date", yaxis_title="Number of cases")
            #Let's make the plot bigger
            #fig.update_layout(height=450, width=1000)
            #fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',paper_bgcolor='rgba(0, 0, 0, 0)',)
            #Let's add somelines to the background
            #fig.update_xaxes(
            #    mirror=True,
            #    ticks='outside',
            #    showline=True,
            #    linecolor='black',
            #    gridcolor='lightgrey'
            #)
            #fig.update_yaxes(
            #    mirror=True,
            #    ticks='outside',
            #    showline=True,
            #    linecolor='black',
            #    gridcolor='lightgrey'
            #)
            #Let's change the message when we hover over the plot and depending if we are hovering over a line or not
            #fig.update_traces(hovertemplate="Date: %{x}<br>Number of: %{y}")
            #Don't show the hover message when we are not hovering over the plot
            #fig.update_layout(hovermode="x") # or "x" or "y" or "x unified" or "y unified"
            #st.plotly_chart(fig)
        #Now let's try to see if we can plot a map with plotly
        #df = px.data.gapminder().query("year==2007")
        #data["ConfirmedCases7Days"] = data.groupby("CountryName")['ConfirmedCases'].rolling(7, center=False).mean().reset_index(0, drop=True)
        #Let's get the confirmedcasses of each country and put it in a new column
        #df["covid"] = data.groupby("CountryName")["ConfirmedCases7Days"].fillna(0).rolling(7, center=False).mean().reset_index(0, drop=True)
        #df["covid"] = df["covid"].fillna(0)
        #Fill inf with 0
        #df["covid"] = df["covid"].replace([np.inf, -np.inf,np.nan], 0)
        # Also can be <NA>, so let's replace it with 0
        #Drop the rows with missing values <NA>
        #fig = px.scatter_geo(df, locations="iso_alpha", color="covid",
        #            hover_name="country", size="covid", 
        #            projection="natural earth")
        #fig.update_layout(height=600, width=1000)
        #Let's update the hover message, to display covid
        #fig.update_traces(hovertemplate="Country: %{hovertext}<br>Number of cases: %{color}")
        #Let's change the color scale
        #fig.update_layout(coloraxis_colorbar=dict(
        #    title="Number of cases",
        #    ))
        #st.plotly_chart(fig)
    if selected == "Computational epimediological models":
        st.markdown("# Computational epidemiological models")
        cols = st.columns((5,2))

        with cols[0]:
            foto1 = Image.open("Foto1.png")
            st.image(foto1)

        with cols[1]:
            st.write("We have developed machine learning-based predictive models of the number"
                    "of hospitalizations and intensive care hospitalizations overall and for"
                    "SARS-CoV-2 patients. We have also developed a model to infer the prevalence"
                    "of the disease based on a few of the answers to our citizen survey "
                    "[https://covid19impactsurvey.org](https://covid19impactsurvey.org/)")




        st.markdown('# Predict cases of Covid-19')

        cols = st.columns((.2,1))
        paises = get_UN_data()
        paises2 = get_UN_data2()
        with cols[0]:
            modes = ["H7","H7 VacW","None","None VacW","XPRIZE","Death predictor"]
            mode = st.selectbox(
                "Select a model ",modes
            )
            paises_list = list(paises.index.unique())
            # Sort the list
            paies_list = sorted(paises_list)
            paises_list2 = list(paises2.index.unique())
            # Sort the list
            paies_list2 = sorted(paises_list2)
            #paises_list.insert(0, "Europe")
            #paises_list.insert(0, "Overall")
            if mode == "H7 VacW":
                country2 = st.selectbox(
                    "Choose countries ",paises_list
                )
            elif mode == "None VacW":
                country2 = st.selectbox(
                    "Choose countries ",paises_list
                )
            else:
                country2 = st.selectbox(
                    "Choose countries ",paises_list2
                )
            
            months_list = ["January","February","March","April","May","June","July","Agost","September","October","November","December"]
            months_dates = ["2020-12-28","2021-01-31","2021-01-31","2021-02-28","2021-02-28","2021-03-31","2021-03-31","2021-04-30",
                            "2021-04-30","2021-05-31","2021-05-31","2021-06-30","2021-06-30","2021-07-31"]
            months_list_short = ["2021_1","2021_2","2021_3","2021_4","2021_5","2021_6","2021_7","2021_8","2021_9","2021_10","2021_11","2021_12"]
            # TODO: Estoy aqui
            month = st.selectbox('Choose a month   ', months_list)
            month = months_list_short[months_list.index(month)]
            if mode == "H7" and (country2 in paises_list):
                # Let's read the file with the predictions
                data = pd.read_csv("latest_predictions/h7_waning_casos/H7_waning_casos_"+month+".csv")
                # Filter by the country
                data = data[data.CountryName == country2].reset_index(drop=True)
                # Group by date  
                data = data.groupby("fecha").mean().reset_index()
            elif mode == "H7 VacW"  and (country2 in paises_list):
                data = pd.read_csv("latest_predictions/h7_waning_casos_vacunas/H7_waning_casos_vacunas_"+month+".csv")
                # Filter by the country
                
                data = data[data.CountryName == country2].reset_index(drop=True)
                
                # Group by date  
                data = data.groupby("fecha").mean().reset_index()
            elif mode == "None" and (country2 in paises_list):
                data = pd.read_csv("latest_predictions/None_waning_casos/None_waning_casos_"+month+".csv")
                # Filter by the country
                data = data[data.CountryName == country2].reset_index(drop=True)
                # Group by date  
                data = data.groupby("fecha").mean().reset_index()
            elif mode == "None VacW" and (country2 in paises_list):
                data = pd.read_csv("latest_predictions/None_waning_casos_vacunas/None_waning_casos_vacunas_"+month+".csv")
                # Filter by the country
                data = data[data.CountryName == country2].reset_index(drop=True)
                # Group by date  
                data = data.groupby("fecha").mean().reset_index()
            elif mode == "XPRIZE":
                data = pd.read_csv("latest_predictions/xprize_all/NONE_xprice_"+month+".csv")
                # Filter by the country
                data = data[data.CountryName == country2].reset_index(drop=True)
                # Group by date  
                data = data.groupby("fecha").mean().reset_index()
            elif mode == "H7" and (country2 in paises_list2):
                data = pd.read_csv("latest_predictions/h7_all/H7_waning_casos_"+month+".csv")
                # Filter by the country
                data = data[data.CountryName == country2].reset_index(drop=True)
                # Group by date  
                data = data.groupby("fecha").mean().reset_index()
            elif mode == None and (country2 in paises_list2):
                data = pd.read_csv("latest_predictions/h7_all/NONE_waning_casos_"+month+".csv")
                # Filter by the country
                data = data[data.CountryName == country2].reset_index(drop=True)
                # Group by date  
                data = data.groupby("fecha").mean().reset_index()
            elif mode == "Death predictor":
                data = pd.read_csv("muertes_predicciones/"+month+".csv")
                st.write(data)
                # Filter by the country
                data = data[data.CountryName == country2].reset_index(drop=True)
                st.write(data)
                # Group by date
                data = data.rename(columns={"Date":"fecha"})
                data = data.groupby("fecha").mean().reset_index()
                # Rename the columns
                data = data.rename(columns={"SmoothNewDeaths":"pred"})
                # Rename the Date by fecha
                
            # Now we plot the data
            with cols[1]:
                fig = go.Figure()
                # Plot the ground truth in orange and dashed
                fig.add_trace(go.Scatter(x=data['fecha'], y=data['truth'], mode='lines', name='Ground truth',line=dict(color='orange', width=4,dash='dash')))
                # Plot the predictions in blue and solid
                fig.add_trace(go.Scatter(x=data['fecha'], y=data['pred'], mode='lines', name='Predictions SVIR',line=dict(color='blue', width=2)))
                # Plot the predictions in blue and solid
                if mode != "Death predictor":
                    fig.add_trace(go.Scatter(x=data['fecha'], y=data['pred_sir'], mode='lines', name='Predictions SIR',line=dict(color='green', width=2)))
                fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20))
                fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',paper_bgcolor='rgba(0, 0, 0, 0)',)
                fig.update_yaxes(
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        linecolor='black',
                        gridcolor='lightgrey'
                    )
                st.plotly_chart(figure_or_data=fig,use_container_width=True)
            
    if selected == "Prescriptor": 
        st.markdown("# Prescriptor Models")
        cols = st.columns((2,5))

        with cols[0]:
            st.write("Our goal in the Prescription phase of the competition is to develop an"
                    "interpretable, data-driven and flexible prescription framework that would"
                    "be usable by non machine-learning experts, such as citizens and policy"
                    "makers in the Valencian Government. Our design principles are therefore"
                    "driven by developing interpretable and transparent models.")

            st.write("Given the intervention costs, it automatically generates up to 10"
                    "Pareto-optimal intervention plans. For each plan, it shows the resulting"
                    "number of cases and overall stringency, its position on the Pareto front"
                    "and the activation regime of each of the 12 types of interventions that"
                    "are part of the plan.")

        with cols[1]:
            foto2 = Image.open("Foto2.png")
            st.image(foto2)

    ############################################################################################################
    #                                         Prescriptor                                                      #
    ############################################################################################################
        st.markdown('# Prescript NPIs for Covid-19')
        cols = st.columns((1,3,3))
        paises = get_UN_data()
        data_pres = pd.read_csv("predictions/robojudge_test.csv")
        #The same as the predictor part
        with cols[0]:
            country_pres = st.selectbox("Choose the country",list(paises.index.unique()))
            data_pres = data_pres[data_pres.CountryName == country_pres]
            data_pres['Date'] = pd.to_datetime(data_pres['Date'], format = '%Y-%m-%d')
            today_pres = min(data_pres.Date)
            start_date_pres = st.date_input('Start date ', today_pres)
            start_date_pres = pd.to_datetime(start_date_pres,format = '%Y-%m-%d')

            tomorrow_pres = max(data_pres.Date)
            end_date_pres = st.date_input('End date ', tomorrow_pres)
            end_date_pres = pd.to_datetime(end_date_pres,format = '%Y-%m-%d')
            if start_date_pres > end_date_pres:
                st.error('Error: End date must fall after start date.')

        
        reg_pres = list(paises.index).count(country_pres)==1
        region = " "
        #regiones = list(paises[paises.index == country_pres].RegionName.fillna(" "))
        
        with cols[0]:
            index = ["Index 0","Index 1","Index 2","Index 3","Index 4",
                    "Index 5","Index 6","Index 7","Index 8","Index 9"]
            value_max = [0,1,2,3,4,5,6,7,8,9]
            index = st.selectbox(
                    "Choose the level of stringency (0-9)", value_max,0
                )

            value_max = [0,1,2,3,4,5,6,7,8,9]
        with cols[1]:
            #Get the data
            prescriptions,stringency = get_prescriptions_and_stringency()
            country_name = country_pres
            #if not reg_pres:
            #    stringency = stringency[(stringency.RegionName == region)]
            cdf = stringency[(stringency['PrescriptorName'] == 'V4C') & (stringency.CountryName == country_pres)]
            #Plotly
            fig = go.Figure()
            #We are going to plot different lines and scatter points, so we use the function add_trace
            #Inside the function we have the data and the type of plot, in this case scatter
            fig.add_trace(go.Scatter(x=cdf['Stringency'],y=cdf['PredictedDailyNewCases'],
                                    name="V4C", 
                                    mode='markers',
                                    marker=dict(size=10,color = 'rgb(29, 126, 235 )',
                                    line=dict(width=1,color='DarkSlateGrey'))))
            # Dont show the legend
            fig.update_layout(showlegend=False)
            #Adittional function to get the pareto front (in the furture we will move it to another file)
            xs, ys = plot_pareto_curve_plotly(list(cdf['Stringency']),list(cdf['PredictedDailyNewCases']))
            #Same thing as before
            fig.add_trace(go.Scatter(x=xs,y=ys, 
                                    mode='lines',
                                    marker=dict(size=10,
                                    color = 'rgb(29, 126, 235 )')))
            #This plot I use it to show the point that we are going to choose
            fig.add_trace(go.Scatter(x=[np.asarray(cdf['Stringency'])[index]],y=[np.asarray(cdf['PredictedDailyNewCases'])[index]],
                                    mode='markers',
                                    marker=dict(size=14,color = 'rgb(255, 0, 0 )'),
                                    line=dict(width=3,color='DarkSlateGrey')))
            #Things to make the plot look better
            fig.update_layout(title='Pareto curve for '+country_name, 
                            xaxis_title='Stringency', 
                            yaxis_title='Predicted Daily New Cases',
                            font=dict(size=12))
            fig.update_layout(
                #margin=dict(l=20, r=20, t=20, b=20),    
                template='seaborn',
                paper_bgcolor='white',
            )
            fig.update_layout(height=450, width=625)
            fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',paper_bgcolor='rgba(0, 0, 0, 0)',)
            #Let's add somelines to the background
            #fig.update_xaxes(
            #    mirror=True,
            #    ticks='outside',
            #    showline=True,
            #    linecolor='black',
            #    gridcolor='lightgrey'
            #)
            fig.update_yaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey'
            )
            
            data_fig1 = fig
            #Finally we plot the figure using st.plotly_chart
            st.plotly_chart(figure_or_data=fig)


        with cols[2]:
            #Same thing as the predictor
            prescription_index = index
            country_name = country_pres
            #C1M_School closing,C2M_Workplace closing,C3M_Cancel public events,C4M_Restrictions on gatherings,C5M_Close public transport,C6M_Stay at home requirements,C7M_Restrictions on internal movement,C8EV_International travel controls,H1_Public information campaigns,H2_Testing policy,H3_Contact tracing,H6M_Facial Coverings
            NPI_COLUMNS = ['C1M_School closing',
                'C2M_Workplace closing',
                'C3M_Cancel public events',
                'C4M_Restrictions on gatherings',
                'C5M_Close public transport',
                'C6M_Stay at home requirements',
                'C7M_Restrictions on internal movement',
                'C8EV_International travel controls',
                'H1_Public information campaigns',
                'H2_Testing policy',
                'H3_Contact tracing',
                'H6M_Facial Coverings']
            region_name = None
            pdf = prescriptions
            #if not reg_pres:
            #    pdf = pdf[(pdf.RegionName == region)]
            #else:
            gdf = pdf[(pdf['PrescriptionIndex'] == prescription_index) &
                        (pdf.CountryName == country_name) &
                        (pdf['RegionName'].isna() if region_name is None else (pdf['RegionName'] == 'region_name'))]
            # gdf tiene que estar entre start_date_pres y end_date_pres 
            gdf["Date"] = pd.to_datetime(gdf["Date"])
            gdf = gdf[(gdf['Date'] >= start_date_pres) & (gdf['Date'] <= end_date_pres)]
            # Comprobamos que la Date y las NPI_COLUMNS sean del mismo tamaño
            if len(gdf) != len(NPI_COLUMNS):
                print("ERROR")
            #Another way to plot plotly, in this case it is called plotly express (See the documentation), the easy plotly
            fig = px.bar(gdf, x ='Date', y=NPI_COLUMNS)
            fig.update_layout(height=450, width=700)
            fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',paper_bgcolor='rgba(0, 0, 0, 0)',)
            #Let's add somelines to the background
            #fig.update_xaxes(
            #    mirror=True,
            #    ticks='outside',
            #    showline=True,
            #    linecolor='black',
            #    gridcolor='lightgrey'
            #)
            fig.update_yaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey'
            )
            fig.update_layout(title='Prescription for '+country_name,  
                            xaxis_title='Date', 
                            yaxis_title='Value',
                            font=dict(size=12))
            # Place the title in the middle
            fig.update_layout(
                #margin=dict(l=20, r=20, t=20, b=20),    
                template='seaborn',
                paper_bgcolor='white',
            )
            data_fig2 = fig
            st.plotly_chart(figure_or_data=fig)
    if selected == "GitHub":
        st.markdown("## GitHub")
        st.markdown("You can find the code of this project in the following link: [GitHub](https://github.com/malozano/covid-xprize )")
        st.markdown("## Comparation with your own model")
        st.markdown("In this section you can compare your model with the one we have developed. You can upload your own predictions and we will compare them with ours." )
        st.markdown("### Upload your own predictions")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            # Can be used wherever a "file-like" object is accepted:
            df = pd.read_csv(uploaded_file)
            st.write(df)
except URLError as e:
    st.error(
        """
            
        **This demo requires internet access.**

        Connection error: %s
        """

        % e.reason
    )
