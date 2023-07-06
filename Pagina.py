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
        options = ["Home", "Computational epidemiological models", "Predictions", "Prescriptor", "Prescriptions", "Data sources", "Team", "GitHub"],
        orientation="horizontal",
        #Let's add some icons
        icons=["house-door", "graph-up","bar-chart-line","graph-up","receipt","file-bar-graph", "people","github"],
        #"receipt","github","newspaper","envelope"],
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
                    \n We have been developing two types of traditional computational epidemiological models: a metapopulation compartmental SEIR model and an agent-based model. However, for this challenge, we opted for a deep learning-based approach, inspired by the model suggested by the challenge organizers. Such an approach would enable us to build a model within the time frame of the competition with two key properties: be applicable to a large number of regions and be able to automatically learn the impact of the Non-Pharmaceutical Interventions (NPIs) on the transmission rate of the disease. The Pandemic COVID-19 XPRIZE challenge has been a great opportunity for our team to explore new modeling approaches and expand our scope beyond the Valencian region of Spain.
                    \n Here, we present a deep learning-based approach to predict the number of daily COVID-19 cases in 30 countries, considering the non-pharmaceutical interventions (NPIs) applied in those countries and including vaccination data of the most used vaccines. With the predictions of new cases, we can prescribe NPIs plans that present the best trade-off between the expected number of COVID-19 cases and the social and economic cost of applying such interventions. This work contributes with an accurate, scalable, data-driven approach to modeling the pandemic curves of cases when vaccination data is available.
                    \n This project has been funded by the Valencian Government, grant VALENCIA IA4COVID (GVA-COVID19/2021/100). Our group also want to acknlowedge previous support from FONDOS
SUPERA COVID-19 Santander-CRUE (CD4COVID19 2020–2021), Fundación BBVA for SARS-CoV-2 research (IA4COVID19 2020-2022) and the Valencian Government.''')
        cols[1].video('https://www.youtube.com/watch?v=RZ9wsSGH8U8')

        cols = st.columns((4))
        cols[0].image("logos/generalitat_valenciana.jpg", width=200)
        cols[1].image("logos/logo_crue.jpg", width=200)
        cols[2].image("logos/logo_santander_universidades.jpg", width=200)
        cols[3].image("logos/logo_bbva.jpg", width=200)
        

    if selected == "Team":
        st.markdown('# Meet The Team')
        cols = st.columns((2))
        with cols[0]:
            st.image("team.jpeg",width=600)
        with cols[1]:
            st.write('''### OUR MULTIDISCIPLINARY TEAM''')
            st.write('''##### VALENCIA IA4COVID''',color = 'yellow')
            st.write('''This group is made up of more than twenty experts from the Universities and research centers of the Valencian Community (Spain) and led by Dr. Nuria Oliver. We have all been working intensively since the beginning of the pandemic, altruistically and using the resources available to us in our respective institutions and with the occasional philanthropic collaboration of some companies.''')
            st.write('''**Affiliated with:**\n
                \n - **Ellis Alicante**: Nuria Oliver.
                \n - **Universitat Politècnica de València**:
                    Alberto Conejero, Miguel Rebollo, Miguel Ángel García March, Òscar Garibo i Orts, Eloy Piñol, Víctor Escaladas, Manuel Portolés,Ahmed Begga, Sergi de María.
                \n - **Universidad de Alicante**:
                    Francisco Escolano, Miguel Ángel Lozano. 
                \n - **Universidad Miguel Hernández**:
                    Kristina Polotskaya, Aurora Mula, Elisa Espín.
                \n - **Universidad Cardenal Herrera CEU**:
                    Antonio Falcó. ''')
        
    if selected == "Computational epidemiological models":
        st.markdown("# Computational epidemiological models")
        cols = st.columns((2))
        with cols[0]:
            foto1 = Image.open("images/v4c.png")
            foto1 = foto1.resize((1000, 500))
            st.image(foto1)
        with cols[1]:
            #st.write("We have developed machine learning-based predictive models of the number"
            #        "of hospitalizations and intensive care hospitalizations overall and for"
            #        "SARS-CoV-2 patients. We have also developed a model to infer the prevalence"
            #        "of the disease based on a few of the answers to our citizen survey "
            #        "[https://covid19impactsurvey.org](https://covid19impactsurvey.org/)")
            st.write('''The Top branch (or context branch) consists of a convolutional layer configured with 64 filters fo size 8 and ReLu as activation function, followed by a bidirectional LSTM layer which encodes the input sequence into states of 64 dimensions, and finally we set a dense layer for prediction. This architecture empirically generalized well for many countries, achieving good performance in both short-term and long-term predictions.
                    \n The Bottom branch (or action branch) consists of an LSTM followed by two dense layers to capture non-linearities. In order to ensure the output is in the $[0,1]$ range we use a sigmoid activation function. In addition, $g(A)$ is constrained so it satisfies the condition: if the difference between two sets of actions $A$ and $A'$ is greater than or equal to 1, then $(1-g(A))$ must be lower or equal to $(1-g(A'))$.''')

        st.write('''We have implemented ten models to be compared. First, we have considered as a baseline model the SIR based model in [1] by M.A. Lozano et al. We point out that we can only include vaccination in a SIR model by adding the H7 Non Pharmaceutial Intervention (Record policies for vaccine delivery for different groups). In order the extend the study of the vaccination effects in the pandemic spread we introduced the vaccination waning modeled by a Weibull distribution function. We have calculated the waning of the protection agains the SARS-CoV-2 due to a previous infection and denoted the resulting models by "waning cases". Additionaly, we have used vaccination data from 30 countries for which we have been able to get the number of vaccines distributed on a daily basis, identified by the vaccine brand. This allows us to compute the waning of the vaccine protection for each vaccine type along time and incorporate this effect to our models. We denote such models by "waning vaccine".  We point out that the "waning cases" have been computed for all the countries in the study and that the countries for which we have computed the "waning vaccine" also include the "waning cases" effect. We consider that the vaccine protection starts 14 after the last vaccine dose (completed or not) and we also consider that people can be reinfected after $d_{0} = 14$ days.''')

        st.header("Weibull distribution")
        st.latex(r'F(n,\lambda_i,k_i)=e^{-(n/\lambda_i)^{k_i}}')
        #st.image("images/weibull.jpg", width=200)
        st.write("where n stands for day and $\lambda_{i}$ and $k_{i}$ are retrieved from the table below.")

        col1, col2 = st.columns((2))
                  
        with col1:
            st.header("Infected people returning to susceptible")
            st.latex(r'\sigma(D)_n^j  =  (1-F(d_0,\lambda_0,k_0))Z_{n-14}')
            st.latex(r'+\sum_{l=1}^{n-d_0}(F(d_0-1+l,\lambda_0,k_0)-F(d_0+l,\lambda_0,k_0))Z_{n-l-14}')
            #st.image("images/waning_cases.jpg", width=400)
            st.write("for $n \ge d_{0} + 1$, and $\lambda_{0} = 87.3$ and $k_{0} = 1.4$.")
        with col2:
            st.header("Vaccinated people returning to susceptible")
            st.latex(r'\gamma(V)_n^j  =  \sum_{v=p,f}\sum_{i=1}^8(1-F(d_0,\lambda_{i,v},k_{i,v}))V^i_{n-14}')
            st.latex(r'+\sum_{v=p,f}\sum_{i=1}^8 \sum_{l=1}^{n-d_1} (F(d_0-1+l,\lambda_{i,v},k_i)-F(d_0+l,\lambda_{i,v},k_{i,v}))V^i_{n-l-14}')
            #st.image("images/waning_vaccine.jpg", width=400)
            st.write("for n $\ge n_{0} + d_{0} = 363$, where $V^i_{s}$ is the number of people vaccinated on the day $s$ with the vaccine $i$, $v$ denotes if people are partially (p) or fully vaccinated (f), and $n_0$ corresponds to December 14th, 2020 (349th day of the year) plus $d_0$ days of latency until people may get infected again when the vaccination started worldwide.")

        col1, col2 = st.columns((2))
        with col1:
            st.header("Considered vaccines:")
            st.write(''' \n  \n  \n
                     \n OA: ChAdOx1 (Oxford/Astrazeneca)
                     \n CA: Ad5-nCoV Convidecia (Cansino)
                     \n MO: mRNA-1273 (Moderna/Biotech)
                     \n SP: BBIBP-CorV (Sinopharm)
                     \n SV: CoronaVac (Sinovac)
                     \n GA: Sputnik V/Gam-COVID-Vac (Gamaleya)
                     \n JA: Ad26.COV2.S (Janssen)
                     \n PB: BNT162b2 (Pfizer/BioNTech)''')
        with col2:
            st.header("Fitted parameters for the Weibull distribution $(\lambda, k)$, for complete and incomplete doses. Values extractec from [2].")
            st.image("images/fitted_parameters.jpg", width=500)

        st.write(''' [1] M.A. Lozano, Ò. Garibo-i Orts, E. Piñol, M. Rebollo, K. Polotskaya,M.A. García-March, J.A. Conejero, F. Escolano, and N. Oliver, ‘Open Data  Science  to  Fight  COVID-19:  Winning  the  500k XPRIZE  Pandemic Response Challenge (Extended Abstract)’, in Proceedings of theThirty-First International Joint Conference on Artificial Intelligence,IJCAI-22, pp. 5304–5308. International Joint Conferences on Artificial Intelligence Organization, (2022). Sister Conferences Best Papers.''')
        st.write(''' [2] C. Hernandez-Suarez and E. Murillo-Zamora, ‘Waning immunity to SARS-CoV-2 following vaccination or infection’,Front. Med.,9,(2022).''')
            
      
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
    if selected == "Predictions":
        #st.markdown("# Computational epidemiological models")
        st.markdown('# Predict cases of Covid-19')

        st.write(''' Two sets of countries have been used to train our models:
                    \n - XPRIZE countries: this set includes all countries in the OxCGRT data set.
                    \n - Vaccination countries: this set includes all countries from which we can extract the number of administered vaccines per day by vaccine type. This set includes the following countries: Argentina, Austria, Belgium, Bulgaria, Canada, Croatia, Cyprus, Czech Republic,  Denmark,  Ecuador,  Estonia,  Finland,  France,  Germany, Hungary, Ireland, Italy, Latvia, Lithuania, Luxembourg, Netherlands, Norway, Poland, Portugal, Slovak Republic, Slovenia, Spain, Sweden, Switzerland, and United States.''')

        st.write(''' We have trained five models to predict new infections by day and one model to predict new deaths by day, with the following particularities:
                    \n - H7 waning cases model. This model is trained with the vaccination countries and includes the H7 NPI and the waning effect of the protection against the SARS-Cov-2 after an individual gets infected.
                    \n - H7 waning vaccine. This model is trained with the vaccination countries and includes the H7 NPI and the waning effect of both the protection against the SARS-Cov-2 after an individual gets infected and after a vaccine dose is administered.
                    \n - No H7 waning cases. This model is trained with the vaccination countries and includes the waning effect of the protection against the SARS-Cov-2 after an individual gets infected.
                    \n - No H7 waning vaccine. This model is trained with the vaccination countries and includes the waning effect of both the protection against the SARS-Cov-2 after an individual gets infected and after a vaccine dose is administered.
                    \n - V4C - XPRIZE waning cases. This model is trained with XPRIZE countries and includes the waning effect of the protection against the SARS-Cov-2 after an individual gets infected and trained with all the available countries in the OxCGRT data set. ''')

        cols = st.columns((.2,1))
        paises = get_UN_data()
        paises2 = get_UN_data2()
        with cols[0]:
            modes = ["H7 waning cases","H7 waning vaccine","No H7 waning cases","No H7 waning vaccine","V4C - XPRIZE"]
            mode = st.selectbox(
                "Select a model ",modes
            )
            paises_list = list(paises.index.unique())
            # Sort the list
            paises_list = sorted(paises_list)
            paises_list2 = list(paises2.index.unique())
            # Sort the list
            paises_list2 = sorted(paises_list2)
            #paises_list.insert(0, "Europe")
            #paises_list.insert(0, "Overall")
            if mode == "H7 waning vaccine":
                country2 = st.selectbox(
                    "Choose countries ",paises_list
                )
            elif mode == "No H7 waning vaccine":
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
            month = st.selectbox('Choose a month in 2021  ', months_list)
            month = months_list_short[months_list.index(month)]
            if mode == "H7 waning cases" and (country2 in paises_list):
                # Let's read the file with the predictions
                data = pd.read_csv("latest_predictions/h7_waning_casos/H7_waning_casos_"+month+".csv")
                # Filter by the country
                data = data[data.CountryName == country2].reset_index(drop=True)
                # Group by date  
                data = data.groupby("fecha").mean().reset_index()
            elif mode == "H7 waning vaccine"  and (country2 in paises_list):
                data = pd.read_csv("latest_predictions/h7_waning_casos_vacunas/H7_waning_casos_vacunas_"+month+".csv")
                # Filter by the country
                
                data = data[data.CountryName == country2].reset_index(drop=True)
                
                # Group by date  
                data = data.groupby("fecha").mean().reset_index()
            elif mode == "No H7 waning cases" and (country2 in paises_list):
                data = pd.read_csv("latest_predictions/None_waning_casos/None_waning_casos_"+month+".csv")
                # Filter by the country
                data = data[data.CountryName == country2].reset_index(drop=True)
                # Group by date  
                data = data.groupby("fecha").mean().reset_index()
            elif mode == "No H7 waning vaccine" and (country2 in paises_list):
                data = pd.read_csv("latest_predictions/None_waning_casos_vacunas/None_waning_casos_vacunas_"+month+".csv")
                # Filter by the country
                data = data[data.CountryName == country2].reset_index(drop=True)
                # Group by date  
                data = data.groupby("fecha").mean().reset_index()
            elif mode == "V4C - XPRIZE":
                data = pd.read_csv("latest_predictions/xprize_all/NONE_xprice_"+month+".csv")
                # Filter by the country
                data = data[data.CountryName == country2].reset_index(drop=True)
                # Group by date  
                data = data.groupby("fecha").mean().reset_index()
            elif mode == "H7 waning cases" and (country2 in paises_list2):
                data = pd.read_csv("latest_predictions/h7_all/H7_waning_casos_"+month+".csv")
                # Filter by the country
                data = data[data.CountryName == country2].reset_index(drop=True)
                # Group by date  
                data = data.groupby("fecha").mean().reset_index()
            elif mode == "No H7 waning cases" and (country2 in paises_list2):
                data = pd.read_csv("latest_predictions/none_all/NONE_waning_casos_"+month+".csv")
                # Filter by the country
                data = data[data.CountryName == country2].reset_index(drop=True)
                # Group by date  
                data = data.groupby("fecha").mean().reset_index()
            
                
            # Now we plot the data
            with cols[1]:
                fig = go.Figure()
                # Plot the ground truth in orange and dashed
                fig.add_trace(go.Scatter(x=data['fecha'], y=data['truth'], mode='lines', name='Ground truth',line=dict(color='orange', width=4,dash='dash')))
                # Plot the predictions in blue and solid
                if mode != "Death predictor":
                    fig.add_trace(go.Scatter(x=data['fecha'], y=data['pred'], mode='lines', name='Predictions SVIR',line=dict(color='blue', width=2)))
                    # Plot the predictions in blue and solid
                
                    fig.add_trace(go.Scatter(x=data['fecha'], y=data['pred_sir'], mode='lines', name='Predictions SIR',line=dict(color='green', width=2)))
                else:
                    fig.add_trace(go.Scatter(x=data['fecha'], y=data['pred'], mode='lines', name='Predicted Deaths',line=dict(color='blue', width=2)))
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
        st.write('''- Death predictor. This model is trained with XPRIZE countries and includes the waning effect of the protection against the SARS-Cov-2 after an individual gets infected and trained with all the available countries in the OxCGRT data set.''')
        cols = st.columns((.2,1))
        paises = get_UN_data()
        paises2 = get_UN_data2()
        with cols[0]:
            modes = ["Death predictor"]
            mode = st.selectbox(
                "Select a model  ",modes
            )
            paises_list = list(paises.index.unique())
            # Sort the list
            paises_list = sorted(paises_list)
            paises_list2 = list(paises2.index.unique())
            # Sort the list
            paises_list2 = sorted(paises_list2)
            #paises_list.insert(0, "Europe")
            #paises_list.insert(0, "Overall")
            if mode == "H7 waning vaccine":
                country2 = st.selectbox(
                    "Choose countries  ",paises_list
                )
            elif mode == "No H7 waning vaccine":
                country2 = st.selectbox(
                    "Choose countries  ",paises_list
                )
            else:
                country2 = st.selectbox(
                    "Choose countries   ",paises_list2
                )
            
            months_list = ["January","February","March","April","May","June","July","Agost","September","October","November","December"]
            months_dates = ["2020-12-28","2021-01-31","2021-01-31","2021-02-28","2021-02-28","2021-03-31","2021-03-31","2021-04-30",
                            "2021-04-30","2021-05-31","2021-05-31","2021-06-30","2021-06-30","2021-07-31"]
            months_list_short = ["2021_1","2021_2","2021_3","2021_4","2021_5","2021_6","2021_7","2021_8","2021_9","2021_10","2021_11","2021_12"]
            # TODO: Estoy aqui
            month = st.selectbox('Choose a month in 2021    ', months_list)
            month = months_list_short[months_list.index(month)]
            
            if mode == "Death predictor":
                data = pd.read_csv("muertes_predicciones/"+month+".csv")
                # Filter by the country
                data = data[data.CountryName == country2].reset_index(drop=True)
                # Group by date
                data = data.rename(columns={"Date":"fecha"})
                data = data.groupby("fecha").mean().reset_index()
                # Rename the columns
                data = data.rename(columns={"SmoothNewDeaths":"pred"})
                # Convert column truth to int
                data['truth'] = data['truth'].astype(int)
                # Rename the Date by fecha
                
            # Now we plot the data
            with cols[1]:
                fig = go.Figure()
                # Plot the ground truth in orange and dashed
                fig.add_trace(go.Scatter(x=data['fecha'], y=data['truth'], mode='lines', name='Ground truth',line=dict(color='orange', width=4,dash='dash')))
                # Plot the predictions in blue and solid
                if mode != "Death predictor":
                    fig.add_trace(go.Scatter(x=data['fecha'], y=data['pred'], mode='lines', name='Predictions SVIR',line=dict(color='blue', width=2)))
                    # Plot the predictions in blue and solid
                
                    fig.add_trace(go.Scatter(x=data['fecha'], y=data['pred_sir'], mode='lines', name='Predictions SIR',line=dict(color='green', width=2)))
                else:
                    fig.add_trace(go.Scatter(x=data['fecha'], y=data['pred'], mode='lines', name='Predicted Deaths',line=dict(color='blue', width=2)))
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

     #   st.write('''###### [1] M.A. Lozano, Ò. Garibo-i Orts, E. Piñol, M. Rebollo, K. Polotskaya,M.A. García-March, J.A. Conejero, F. Escolano, and N. Oliver, ‘OpenData  Science  to  Fight  COVID-19:  Winning  the  500k XPRIZE  Pandemic Response Challenge (Extended Abstract)’, in Proceedings of theThirty-First International Joint Conference on Artificial Intelligence,IJCAI-22, pp. 5304–5308. International Joint Conferences on ArtificialIntelligence Organization, (2022). Sister Conferences Best Papers.''')
            
    if selected == "Prescriptor": 
        st.markdown("# Prescriptor Models")
        cols = st.columns((2))

        with cols[0]:
            st.write("Our goal in the Prescription phase of the competition is to develop an interpretable, data-driven and flexible prescription framework that would be usable by non machine-learning experts, such as citizens and policy makers in the Valencian Government. Our design principles are therefore driven by developing interpretable and transparent models.")

            st.write("Given the intervention costs, it automatically generates up to 10 Pareto-optimal intervention plans. For each plan, it shows the resulting number of cases and overall stringency, its position on the Pareto front and the activation regime of each of the 12 types of interventions that are part of the plan.")

        with cols[1]:
            foto2 = Image.open("images/Prescriptor.png")
            # cambiamos el tamaño de la imagen
            foto2 = foto2.resize((600, 400))
            st.image(foto2)

    ############################################################################################################
    #                                         Prescriptor                                                      #
    ############################################################################################################
    if selected == "Prescriptions": 
        st.markdown('# Prescript NPIs for Covid-19')
        st.write(''' The Pareto front for a given country, shows the expected number of cases derived from the application of different NPIs plans that result in different stringency levels. Different stringencies result in a different number of cases. The Prescriptor shows the NPI plan which fits the stringency while minimizing the economic costs derived from the appication of such NPIs plan, using the costs for each NPI which were published in [3] and that we show in Table 2 in the Data sources section.''')
        cols = st.columns((1,3,3))
        paises = get_UN_data()
        data_pres = pd.read_csv("predictions/robojudge_test.csv")
        #The same as the predictor part
        with cols[0]:
            # Ponemos por defecto España en el selectbox
            country_pres = st.selectbox("Choose the country",sorted(list(paises.index.unique())),index=sorted(list(paises.index.unique())).index("Spain"))
            data_pres = data_pres[data_pres.CountryName == country_pres]
            data_pres['Date'] = pd.to_datetime(data_pres['Date'], format = '%Y-%m-%d')
            today_pres = min(data_pres.Date)
            # Ponemos por defecto la fecha del 01-01-2021
            min_date = pd.to_datetime("2021-01-01",format = '%Y-%m-%d')
            start_date_pres = st.date_input('Start date ', min_date)
            start_date_pres = pd.to_datetime(start_date_pres,format = '%Y-%m-%d')

            tomorrow_pres = max(data_pres.Date)
            max_date = pd.to_datetime("2021-01-31",format = '%Y-%m-%d')
            end_date_pres = st.date_input('End date ', max_date)
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
                    "Choose the level of stringency (0-9)", value_max,7
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
            fig.update_layout(title='Pareto front for '+country_name, 
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
            NPI_COLUMNS = ['C1: School closing',
                'C2: Workplace closing',
                'C3: Cancel public events',
                'C4: Restrictions on gatherings',
                'C5: Close public transport',
                'C6: Stay at home requirements',
                'C7: Restrictions on internal movement',
                'C8: International travel controls',
                'H1: Public information campaigns',
                'H2: Testing policy',
                'H3: Contact tracing',
                'H6: Facial Coverings']
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
            # cambiamos los nombres de las columnas a los nuevos que tenemos en NPI_COLUMNS
            gdf = gdf.rename(columns={"C1M_School closing":"C1: School closing",
                                    "C2M_Workplace closing":"C2: Workplace closing",
                                    "C3M_Cancel public events":"C3: Cancel public events",
                                    "C4M_Restrictions on gatherings":"C4: Restrictions on gatherings",
                                    "C5M_Close public transport":"C5: Close public transport",
                                    "C6M_Stay at home requirements":"C6: Stay at home requirements",
                                    "C7M_Restrictions on internal movement":"C7: Restrictions on internal movement",
                                    "C8EV_International travel controls":"C8: International travel controls",
                                    "H1_Public information campaigns":"H1: Public information campaigns",
                                    "H2_Testing policy":"H2: Testing policy",
                                    "H3_Contact tracing":"H3: Contact tracing",
                                    "H6M_Facial Coverings":"H6: Facial Coverings"})
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

        st.write('In order to check the meaning of each NPI level, please check the table below.')
    ############################################################################################################
        ids = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'H1', 'H2', 'H3', 'H6']
        description = ['Record closings of schools and universities', 'Record closings of workplaces', 'Record cancelling public events', 'Record limits on gatherings', 'Record closing of public transport','Record orders to "shelter-in-place" and otherwise confine to the home', 'Record restrictions on internal movement between cities/regions', 'Record restrictions on international travel',
                       'Record presence of public info campaigns', 'Record government policy on who has access to testing', 'Record government policy on contact tracing', 'Record policies on the use of facial coverings outside the home']
        coding = ['0 - no measures \n 1 - recommend closing or all schools open with alterations resulting in significant differences compared to non-Covid-19 operations \n 2 - require closing (only some levels or categories, eg just high school, or just public schools) \n 3 - require closing all levels \n Blank - no data',
                  '0 - no measures \n 1 - recommend closing (or recommend work from home) or all businesses open with alterations resulting in significant differences compared to non-Covid-19 operation \n 2 - require closing (or work from home) for some sectors or categories of workers \n 3 - require closing (or work from home) for all-but-essential workplaces (eg grocery stores, doctors) \n Blank - no data',
                  '0 - no measures \n 1 - recommend cancelling \n 2 - require cancelling \n Blank - no data',
                  '0 - no restrictions \n 1 - restrictions on very large gatherings (the limit is above 1000 people) \n 2 - restrictions on gatherings between 101-1000 people \n 3 - restrictions on gatherings between 11-100 people \n 4 - restrictions on gatherings of 10 people or less \n Blank - no data',
                  '0 - no measures \n 1 - recommend closing (or significantly reduce volume/route/means of transport available) \n 2 - require closing (or prohibit most citizens from using it) \n Blank - no data',
                  '0 - no measures \n 1 - recommend not leaving house \n 2 - require not leaving house with exceptions for daily exercise, grocery shopping, and ‘essential’ trips \n 3 - require not leaving house with minimal exceptions (eg allowed to leave once a week, or only one person can leave at a time, etc) \n Blank - no data',
                  '0 - no measures \n 1 - internal movement restrictions in place (e.g. strict restrictions in some regions/cities) \n 2 - internal movement restrictions recommended (e.g. “reduced” mobility) \n 3 - no internal movement restrictions \n Blank - no data',
                  '0 - no restrictions \n 1 - screening arrivals \n 2 - quarantine arrivals from some or all regions \n 3 - ban arrivals from some regions \n 4 - ban on all regions or total border closure \n Blank - no data',
                  '0 - no Covid-19 public info campaign \n 1 - public officials urging caution about Covid-19 \n 2 - coordinated public info campaign (eg across traditional and social media) \n Blank - no data',
                  '0 - no testing policy \n 1 - only those who both (a) have symptoms AND (b) meet specific criteria (eg key workers, admitted to hospital, came into contact with a known case, returned from overseas) \n 2 - testing of anyone showing Covid-19 symptoms \n 3 - open public testing (eg “drive through” testing available to asymptomatic people) \n Blank - no data',
                  '0 - no contact tracing \n 1 - limited contact tracing; not done for all cases \n 2 - comprehensive contact tracing; done for all identified cases \n Blank - no data',
                    '0 - no policy \n 1 - recommended \n 2 - required in some specified shared/public spaces outside the home with other people present and not able to maintain physical distancing \n 3 - required outside the home at all times regardless of location or presence of other people \n Blank - no data']
        # creaos la misma lista que antes, pero ahora añadiremos saltos de linea despues de cada numero
        # quitamos el Blank - no data de la lista
        coding = [x.replace('Blank - no data', '') for x in coding]
        df = pd.DataFrame()
        df['NPI'] = ids
        df['Description'] = description
        df['Coding'] = coding
        df = df.set_index('NPI')
        # alineamos el texto de las columnas al centro
        st.dataframe(df.style.set_properties(**{'text-align': 'center'}),
                    width=1600)
        #st.table(df)
        st.write(''' [3] V.  Janko,  N.  Reščič,  A.  Vodopija,  D.  Susič,  C.  De  Masi,  T.  Tušar,A. Gradišek, S. Vandepitte, D. De Smedt, J. Javornik, M. Gams, andM.  Luštrek,  ‘Optimizing  non-pharmaceutical  intervention  strategies against COVID-19 using artificial intelligence’,Front. Pub. Health,11,(2023).''') 
        
    ############################################################################################################
    if selected == "GitHub":
        st.markdown("## GitHub")
        st.markdown("You can find the code of this project in the following link: [GitHub](https://github.com/AhmedBegggaUA/V4C)")
        #st.markdown("## Comparation with your own model")
        #st.markdown("In this section you can compare your model with the one we have developed. You can upload your own predictions and we will compare them with ours." )
        #st.markdown("### Upload your own predictions")
        #uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        #if uploaded_file is not None:
        #    # Can be used wherever a "file-like" object is accepted:
        #    df = pd.read_csv(uploaded_file)
        #    st.write(df)
    if selected == "Data sources":
        st.markdown("## Data sources")
        
        st.write('We have retrieved the number of infected and vaccinated people and the non-pharmaceutical interventions (NPIs) applied in each country of interest from the Oxford Covid-19 Government Response Tracker (OxCGRT) [3]. In Table 1 we depict the considered NPIs. NPIs are categorical variables whose value indicates the application level set for each country, being the higher the level, the more restrictive the measure is applied. H7 NPI indicates if vaccines were not available (0), if they were available to some particular groups among key workers,clinically vulnerable groups, and elderly groups (1-3), broader groups (4) or universally available (5). In [4] a full description for each NPI can be found. The prediction models consider confinement NPIs (C1 to C8) and some public health interventions (H1 to H3 and H6). We use the H/ NPI to incorporate vaccination in a SIR model or complement an SVIR model.')

        st.write('From the OxCGRT data set we also retrieve the number of administered vaccine doses per country and day, including the information of which vaccines were used in each day. But the number of vaccines is not split per vaccine type. The number of vaccine doses in each day by vaccine type can be retrieved from [6, 7], where it is gathered for the following list of countries: Argentina, Austria, Belgium, Bulgaria, Canada, Croatia, Cyprus, Czech Republic,  Denmark,  Ecuador,  Estonia,  Finland,  France,  Germany, Hungary, Ireland, Italy, Latvia, Lithuania, Luxembourg, Netherlands, Norway, Poland, Portugal, Slovak Republic, Slovenia, Spain, Sweden, Switzerland, and United States. From this data set we compute de % of administered vaccines by day and by vaccine type, using these daily % to compute the number of administered vaccines by country and day using the OxCGRT data set.')

        cols = st.columns((2))
        with cols[0]:
            st.write('#### Table 1. Considered NPIs and their activation values')
            _npis = ['C1. School closing', 'C2. Workplace closing', 'C3. Cancelation of public events', 'C4. Restrictions on gatherings','C5. Close public transport', 'C6. Stay at home requirements', 'C7. Internal movement restrictions', 
                        'C8. Intl. travel controls', 'H1. Public info. campaigns', 'H2. Testing policy', 'H3. Contact tracing','H6. Facial coverings', 'H7. Vaccination policy']
            _values = ['[0,1,2,3]', '[0,1,2,3]', '[0,1,2]', '[0,1,2,3]', '[0,1,2]', '[0,1,2,3]', '[0,1,2]', '[0,1,2,3]','[0,1,2]', '[0,1,2,3]', '[0,1,2]', '[0,1,2,3,4]', '[0,1,2,3,4,5]']
            df2 = pd.DataFrame()
            df2['NPI name'] = _npis
            df2['Values'] = _values
            df2 = df2.set_index('NPI name')
            # alineamos el texto de las columnas al centro
            
            # Aliniamos el texto de las filas a la izquierda
            st.dataframe(df2.style.set_properties(**{'text-align': 'center'}),width=900)

        with cols[1]:
            st.write('#### Table 2. Economic costs as % of the GDP loss if NPI applied')
            _economic = [3.9, 22.0, 1.4, 1.4, 0.3, 5.2, 7.8, 6.6, 0.0026, 0.6, 0.1, 0.03]
            _social = [11, 11, 7, 10, 2, 12, 10, 2, 1, 1, 1, 5]
            _combined = [0.55, 0.96, 0.32, 0.45, 0.09, 0.62, 0.59, 0.20, 0.04, 0.05, 0.04, 0.21]
            df = pd.DataFrame()
            df['NPI name'] = _npis[:-1]
            df['Economic'] = _economic
            df['Social'] = _social
            df['Combined'] = _combined
            df = df.set_index('NPI name')
            # POnemos la tabla centrada
            st.dataframe(df.style.set_properties(**{'text-align': 'center'}),width=900)

        st.write(''' [3] V.  Janko,  N.  Reščič,  A.  Vodopija,  D.  Susič,  C.  De  Masi,  T.  Tušar,A. Gradišek, S. Vandepitte, D. De Smedt, J. Javornik, M. Gams, andM.  Luštrek,  ‘Optimizing  non-pharmaceutical  intervention  strategies against COVID-19 using artificial intelligence’,Front. Pub. Health,11,(2023).''') 
        st.write(''' [4]  T. Hale, N. Angrist, R. Goldszmidt, B. Kira, A. Petherick, T. Phillips,S. Webster, E. Cameron-Blake, L. Hallas, S. Majumdar, et al., ‘A global panel database of pandemic policies (Oxford COVID-19 Government Response Tracker)’,Nat. Hum. Behav.,5(4), 529–538, (2021).''')
        st.write(''' [5] T.Hale,N.Angrist,R.Goldszmidt,B.Kira,A.Peth-erick,etal.COVID-19 Government Response Tracker.https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md, 2021.''')       
        st.write(''' [6] E. Mathieu, H. Ritchie, E. Ortiz-Ospina, M. Roser, J. Hasell, C. Ap-pel,  C.  Giattino,  and  L.  Rodés-Guirao.   Data  on  COVID-19  (coronavirus) vaccinations by Our World in Data.   https://github.com/owid/covid-19-data/tree/master/public/data/vaccinations, 2021.  [Online; ac-cessed 30-April-2023].''')
        st.write(''' [7] E. Mathieu, H. Ritchie, E. Ortiz-Ospina, M. Roser, J. Hasell, C. Appel,C.  Giattino,  and  L.  Rodés-Guirao,  ‘A  global  database  of  COVID-19 vaccinations’,Nature Hum. Behav.,5(7), 947–953, (2021).''')

except URLError as e:
    st.error(
        """
            
        **This demo requires internet access.**

        Connection error: %s
        """

        % e.reason
    )
