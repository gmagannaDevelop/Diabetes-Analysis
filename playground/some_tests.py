!ls | grep py
from preproc import import_csv
x = import_csv("data_test/NG1988812H_Maganna_Gustavo_13-02-20_6-05-20.csv")
x
type(x.index)
set(x["New Device Time"])
x["New Device Time"].unique
x["New Device Time"].unique()
x = import_csv("data_test/NG1988812H_Maganna_Gustavo_13-02-20_6-05-20.csv")
x[x.columns[31]]
x[x.columns[31]].unique()
x.columns[31]
x[x.columns[36]].unique()
x.columns[36]
x[x.columns[41]].unique()
x.columns[41]
 x
x["Sensor Glucose (mg/dL)"]
y = pd.DataFrame({"x": range(35)})
y
y = pd.DataFrame({"x": range(1,36)})
y.diff(2)
y.diff(10)
x
x['Sensor Glucose (mg/dL)']
x['Sensor Glucose (mg/dL)'].plot()
plt.show()
plt.style.use("seaborn")
plt.show()
x['Sensor Glucose (mg/dL)'].plot(); plt.show()
x.loc["2020-03-03", 'Sensor Glucose (mg/dL)'].plot(); plt.show()
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].plot(); plt.show()
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].interpolate()
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].plot(); plt.show()
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].plot()
plt.close("all")
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].interpolate().plot(**{"label": "interpolated"})
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].plot(**{"label": "original"})
plt.legend()
plt.show()
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].resample("1M").interpolate()
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].resample("1T").interpolate()
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].resample("1T")
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].resample("1T").fillna("0")
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].resample("1T").interpolate()
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].interpolate().plot(**{"label": "interpolated"})
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].resample("1T").interpolate().plot(**{"label": "resampled"})
plt.close("all")
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].resample("1T").interpolate().plot(**{"label": "resampled"})
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].interpolate().plot(**{"label": "interpolated"})
x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].plot(**{"label": "original"})
plt.legend()
plt.show()
%history -f testies
