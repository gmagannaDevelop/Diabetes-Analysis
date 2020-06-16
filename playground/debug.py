
from preproc import import_csv


x = import_csv("data_test/NG1988812H_Maganna_Gustavo_(13-02-20)_(6-05-20).csv")

plt.style.use("seaborn")

d = {
    "resampled": x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].resample("1T").interpolate(),
    "interpolated": x.loc["2020-04-03", 'Sensor Glucose (mg/dL)'].interpolate(),
    "original": x.loc["2020-04-03", 'Sensor Glucose (mg/dL)']
}
#labeledplot = lambda obj, label: obj.plot(**{"label": f"{label}"})

#for label, obj in d.items():
#    labeledplot(obj, label)
d["resampled"].plot(label="glycaemia")
d["resampled"].diff(30).plot(label="30 min")
d["resampled"].diff(120).plot(label="2 hours")

plt.legend()
plt.show()


