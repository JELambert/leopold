import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

to_update1 = "update1" + ".csv"
to_update2 = "update2" + ".csv"

now = "5/26/21 -- 2:00 pm CST"
def fix_vote(x):
    if x['Custom_Variable3'] == "VOTED":
        x['voted'] = "YES"
    else:
        x['voted'] = x['voted']
#@st.cache
def load_data(to_update1, to_update2):
    d1 = pd.read_csv(to_update1)
    d2 = pd.read_csv(to_update2)
    df = pd.concat([d1,d2])
    df['Favorability'] = df['Favorability'].str.replace("FAVOR", "Favor").str.replace("UNDECIDED", "Undecided")
    df['voted'] = df['voted'].str.replace("Yes","YES").str.replace("NO", "No")
    df['voted'] = df['voted'].fillna("No")
    #df['voted'] = df.apply(fix_vote,axis=1)
    return df

df = load_data(to_update1, to_update2)
favornew = df.loc[(df.Favorability == "Favor") | (df.Favorability == "Not Favor")]



def get_turnout(favornew):
    favors = favornew.loc[favornew.Favorability=="Favor"].groupby("Jurisdiction_Precinct")['Favorability'].count().reset_index()
    nons = favornew.loc[favornew.Favorability!="Favor"].groupby("Jurisdiction_Precinct")['Favorability'].count().reset_index()
    m = favors.merge(nons, on='Jurisdiction_Precinct', how='outer').rename(columns={'Favorability_x':'Favorables', 'Favorability_y':'NonFavs'})
    turnout = pd.read_csv('turnout.csv')
    turnout['percentg2'] = turnout['percentg']*100
    tangifinal = turnout.merge(m, left_on='precinct', right_on='Jurisdiction_Precinct', how='left', indicator=True)
    return tangifinal

turnout = get_turnout(favornew)


def mweight(x):

    weight = 1- ((x['turnout'] - (x['Favorables'] + x['NonFavs']))/x['turnout'])


    return weight

def make_known(x):
    try:

        known = x['weight'] * ( x['Favorables'] / (x['Favorables'] + x['NonFavs']))

        return known

    except ZeroDivisionError:
        return 0

def make_predicted(x):

    predicted = (1-x['weight']) * (x['percentg'])
    return predicted


st.sidebar.title("Leopold Predictions")
page = st.sidebar.radio(
     "Pick an option",
     ('Predictions', 'Demogaphics', 'Favorable Analysis'),
     )


if page =="Predictions":
    st.title("Predictions")

    st.markdown("Last Updated: " + now)
    st.subheader("Baseline Predictions")
    t = "22,787"
    p = f'{round(len(df), 0):,}'
    st.markdown("The baseline predicted turnout number is **" + t + "** out of a total possible voting base of **" + p + "**")
    #g = str(round(turnout[' grace '].sum(), 0))
    #a = str(round(turnout.turnout.sum() - turnout[' grace '].sum(), 0))
    g = "3,044"
    st.markdown("Our baseline prediction for total votes before identifying favorables is **" + g + "**")

    #######
    rad = st.radio(
         "Pick a Forecast",
         ('Smart', 'In-Depth'),
         )

    tangifinal = turnout.copy()

    tangifinal[['Favorables', 'NonFavs']] = tangifinal[['Favorables', 'NonFavs']].fillna(0)
    tangifinal['weight'] = tangifinal.apply(mweight, axis=1)
    tangifinal['known'] = tangifinal.apply(make_known, axis=1)
    tangifinal['predicted'] = tangifinal.apply(make_predicted, axis=1)
    tangifinal['pr'] = tangifinal.predicted + tangifinal.known

    st.subheader("WORK IN PROGRESS")
    st.subheader("________________")
    if rad=="Smartt":


        st.subheader("SMART Forecast")
        st.markdown("*Forecast is based on a weighted model. Weights are determinend by previous turnout at precinct levels versus the number of favorables/nonfavorables identifed.*")
        st.markdown("Adjust Turout percentage to garner total vote values.")
        x = st.slider('Baseline turnout percent is 54%', value=54)
        tangifinal.votes = (tangifinal['pr']-.0212) * tangifinal['turnout'] * (x/54)
        tangifinal.vagainst = (1- (tangifinal['pr']-.0212)) * tangifinal['turnout'] * (x/54)


        vf = round(tangifinal.votes.sum(), 0)
        va = round(tangifinal.vagainst.sum(), 0)
        total = (vf / (vf +va) ) *100
        st.markdown("Our adjusted prediction for total votes *after* phonebanking is **" + str(vf) + "** compared to the against total of **" + str(va) + "**")

        if round(tangifinal.votes.sum(), 0) > round(tangifinal.vagainst.sum(), 0):
            st.subheader("The forecast predicts a WIN at " + str(round(total,2)) + "%")
        else:
            st.subheader("Sadly the forecast predicts a LOSS:(")


    elif rad=="In-Deptht":

        st.header("Build-A-Forecast (Individual Precincts)")
        t2 = tangifinal.copy()

        case = st.radio(
             "Pick a confidence level",
             ('Worst Case', 'Moderate', 'Optimal'),
             )

        if case == "Worst Case":
            t2['new'] = t2['trump results']
            t2['pr'] = 1 - t2['new']
            t2['pr'] = t2['pr'].astype('float')
            #t2['pr'] = t2['pr']-.08
        elif case == "Moderate":
            t2['pr'] = t2['pr']-.065333
        elif case == "Optimal":
            t2['pr'] = t2['pr'] -.0212

        precincts = sorted(t2.precinct.unique())

        temp_votes = []
        tot_votes = []
        col1, col2 = st.beta_columns(2)
        with col1:
            count = 0
            for prec in precincts:

                st.subheader("Precinct: " + prec)
                predic = t2.loc[t2['precinct']==prec]['pr'].values[0]

                ttl = "Adjust percent for Grace, (Baseline is: " + str(round(predic*100, 2)) + "%)"

                x_040a = st.slider(ttl,min_value=float(0), max_value=float(100), value=float(round(predic*100, 2)), key=count)

                ttl2 = 'Adjust precinct Turnout ' + prec+ ", (Baseline turnout is: " + str(round(t2.loc[t2['precinct']==prec]['turnout'].values[0], 2))
                minv = float(round(t2.loc[t2['precinct']==prec]['turnout'].values[0]/2, 0))
                maxv = float(round(t2.loc[t2['precinct']==prec]['turnout'].values[0]*2, 0))
                v = float(round(t2.loc[t2['precinct']==prec]['turnout'].values[0], 0))
                x_040b = st.slider(ttl2,min_value=minv, max_value=maxv, value=v)

                st.write("Votes for GRACE are " + str(round((x_040a/100) * x_040b, 0)))
                temp_votes.append(round((x_040a/100) * x_040b, 0))
                tot_votes.append(round((1-(x_040a/100)) * x_040b, 0))
                count += 1
                st.markdown("----")


        with col2:
            for prec in precincts:

                tfor = round(sum(temp_votes), 0)
                taga = round(sum(tot_votes), 0)
                total = ( tfor / (tfor + taga) )*100
                if tfor>taga:
                    st.header("WIN")
                else:
                    st.header("LOSS:(")

                st.markdown("The total votes for GRACE are: **" + str(round(sum(temp_votes), 0)) + "**")
                st.markdown("The total votes against GRACE are: **" + str(round(sum(tot_votes), 0)) + "**")

                st.markdown("The percent for GRACE is: **" + str(round(total, 2)) + "** %" )
                st.markdown("---")
                st.markdown("---")
                st.markdown("---")
                st.markdown("---")



elif page=="Demogaphics":

    st.title("Demographics Page")

    df.rename(columns={'Jurisdiction_Precinct': 'Precinct', 'Personal_Race':'Race', 'Personal_Sex':'Sex', 'Personal_Age':'Age', 'Registration_PoliticalPartyCode':'Party'}, inplace=True)

    st.subheader("Demographic Descriptives")
    st.write(pd.crosstab(df['Race'], df['Sex']))


    st.write(pd.crosstab(df['Party'], df['Sex']))
    favs = len(df.loc[df.Favorability=="Favor"])
    favsv = len(df.loc[(df.voted == "YES") & (df.Favorability == "Favor")])
    voted = favsv/favs *100
    #st.markdown("Percentage of favorables who have voted **" + str(round(voted, 2)) + "** %")


    df['Precinct'] = df['Precinct'].str.lstrip("0")
    df = df.dropna(subset=['Precinct'])



elif page=="Favorable Analysis":

    df.rename(columns={'Jurisdiction_Precinct': 'Precinct', 'Personal_Race':'Race', 'Personal_Sex':'Sex', 'Personal_Age':'Age', 'Registration_PoliticalPartyCode':'Party'}, inplace=True)

    st.header("Favorable Analysis")

    st.markdown("Total Number of Favorables: **" + str(len(df.loc[df.Favorability=="Favor"])) + "**" )

    df = df.dropna(subset=['Precinct'])

    col1, col2 = st.beta_columns(2)
    with col1:
        for col in ['Rating', 'Precinct', 'Race',  'Sex', 'Party']:
            st.subheader(col)
            if col =='Age':
                ratin = list(df[col].unique())
            else:

                ratin = list(df[col].astype('str').unique())

            stt = "which" + col
            ra = st.multiselect(col,sorted(ratin))

            prra = df.loc[df[col].isin(ra)]
            favsprra = len(prra.loc[prra.Favorability=="Favor"])
            favsvprra = len(prra.loc[(prra.voted == "YES") & (prra.Favorability == "Favor")])
            if favsprra == 0:
                votedprra = 0
            else:
                votedprra = favsprra/len(df.loc[df.Favorability=="Favor"]) *100

            st.markdown("Percentage of favorables in this group is **" + str(round(votedprra, 2)) + "** %")
            st.markdown(str(favsvprra) + " / " + str(favsprra))
            st.markdown("---")

    with col2:

        st.subheader("Age")
        valuesa = st.slider(
        'Select a range of Age Values',
        18.0, 132.0, (24.0, 75.0))

        adf = df.loc[(df['Age'] >= valuesa[0]) & (df['Age']<=valuesa[1])]
        fage = len(adf.loc[adf.Favorability=="Favor"])
        fage2 = len(adf.loc[(adf.voted == "YES") & (adf.Favorability == "Favor")])

        if fage == 0:
            votedprra = 0
        else:
            votedprra = fage/len(df.loc[df.Favorability=="Favor"]) *100

        st.markdown("Percentage of favorables in this group is **" + str(round(votedprra, 2)) + "** %")
        st.markdown(str(fage) + " / " + str(len(df.loc[df.Favorability=="Favor"])))
        st.markdown("---")


        num_m = st.number_input('Age Minimum', min_value=float(18.0), max_value=float(132.0), value=float(18.0), step=float(1))
        num_ma = st.number_input('Age Maximum', min_value=float(18.0), max_value=float(132.0), value=float(18.0), step=float(1))



        adf = df.loc[(df['Age'] >= num_m) & (df['Age']<=num_ma)]
        fage = len(adf.loc[adf.Favorability=="Favor"])
        fage2 = len(adf.loc[(adf.voted == "YES") & (adf.Favorability == "Favor")])

        if fage == 0:
            votedprra = 0
        else:
            votedprra = fage/len(df.loc[df.Favorability=="Favor"]) *100

        st.markdown("Percentage of favorables in this group is **" + str(round(votedprra, 2)) + "** %")
        st.markdown(str(fage) + " / " + str(len(df.loc[df.Favorability=="Favor"])))
        st.markdown("---")


elif page=="EXTRA":
        fig, ax = plt.subplots()

        l = sns.countplot(
            y="Age", hue="Sex",
             palette="dark",ax=ax,data=df,)
        l.tick_params(labelsize=5)

        st.pyplot()
