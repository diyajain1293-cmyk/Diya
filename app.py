
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Employee Attrition Intelligence", layout="wide")
st.title("Employee Attrition Intelligence Dashboard")
st.caption("Interactive analytics + ML for better retention decisions")

def load_data(file):
    if file is None:
        st.info("Upload your dataset (Excel/CSV) to begin.")
        return None
    name = file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df

def make_binary_target(series):
    s = series.copy()
    if s.dtype == "O" or str(s.dtype).startswith("category"):
        s = s.astype(str).str.strip()
        if {"Yes","No"}.issubset(set(s.unique())):
            return (s=="Yes").astype(int)
        return s.str.lower().isin(["yes","y","true","1"]).astype(int)
    return (s.astype(float)>0).astype(int)

def build_preprocess(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ])
    return preprocess, num_cols, cat_cols

def apply_filters(df, jobrole_list, satisfaction_col, sat_min):
    d = df.copy()
    if jobrole_list:
        if "JobRole" in d.columns:
            d = d[d["JobRole"].isin(jobrole_list)]
    if satisfaction_col and satisfaction_col in d.columns:
        try:
            d = d[d[satisfaction_col] >= sat_min]
        except Exception:
            pass
    return d

def oof_eval(pipe, X, y, n_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_oof_pred = np.zeros_like(y, dtype=int)
    y_oof_proba = np.zeros_like(y, dtype=float)

    train_acc = []; train_prec=[]; train_rec=[]; train_f1=[]; train_auc=[]
    test_acc  = []; test_prec=[]; test_rec=[]; test_f1=[]; test_auc=[]

    for tr, te in cv.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        pipe.fit(Xtr, ytr)

        ytr_pred = pipe.predict(Xtr)
        try:
            ytr_proba = pipe.predict_proba(Xtr)[:,1]
        except Exception:
            dec = pipe.decision_function(Xtr)
            dec_min, dec_max = dec.min(), dec.max()
            ytr_proba = (dec - dec_min)/(dec_max-dec_min) if dec_max>dec_min else np.zeros_like(dec, dtype=float)

        train_acc.append(accuracy_score(ytr, ytr_pred))
        train_prec.append(precision_score(ytr, ytr_pred, zero_division=0))
        train_rec.append(recall_score(ytr, ytr_pred, zero_division=0))
        train_f1.append(f1_score(ytr, ytr_pred, zero_division=0))
        train_auc.append(roc_auc_score(ytr, ytr_proba))

        yte_pred = pipe.predict(Xte)
        try:
            yte_proba = pipe.predict_proba(Xte)[:,1]
        except Exception:
            dec = pipe.decision_function(Xte)
            dec_min, dec_max = dec.min(), dec.max()
            yte_proba = (dec - dec_min)/(dec_max-dec_min) if dec_max>dec_min else np.zeros_like(dec, dtype=float)

        y_oof_pred[te] = yte_pred
        y_oof_proba[te] = yte_proba

        test_acc.append(accuracy_score(yte, yte_pred))
        test_prec.append(precision_score(yte, yte_pred, zero_division=0))
        test_rec.append(recall_score(yte, yte_pred, zero_division=0))
        test_f1.append(f1_score(yte, yte_pred, zero_division=0))
        test_auc.append(roc_auc_score(yte, yte_proba))

    cm = confusion_matrix(y, y_oof_pred)
    fpr, tpr, _ = roc_curve(y, y_oof_proba, pos_label=1)
    auc_val = roc_auc_score(y, y_oof_proba)
    metrics = {
        "Train Accuracy (mean)": np.mean(train_acc),
        "Test Accuracy (mean)": np.mean(test_acc),
        "Train Precision (mean)": np.mean(train_prec),
        "Test Precision (mean)": np.mean(test_prec),
        "Train Recall (mean)": np.mean(train_rec),
        "Test Recall (mean)": np.mean(test_rec),
        "Train F1 (mean)": np.mean(train_f1),
        "Test F1 (mean)": np.mean(test_f1),
        "Train ROC AUC (mean)": np.mean(train_auc),
        "Test ROC AUC (mean)": np.mean(test_auc),
    }
    return metrics, cm, fpr, tpr, auc_val

def plot_bw_confusion(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_xlim(0,2); ax.set_ylim(0,2)
    for x in range(3):
        ax.plot([x,x],[0,2], color="black", linewidth=1)
        ax.plot([0,2],[x,x], color="black", linewidth=1)
    for i in range(2):
        for j in range(2):
            ax.text(j+0.5, 1.5-i, str(cm[i,j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0.5,1.5]); ax.set_xticklabels(["Stay (0)","Leave (1)"])
    ax.set_yticks([0.5,1.5]); ax.set_yticklabels(["Stay (0)","Leave (1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title); ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)

def combined_roc_plot(roc_dict):
    fig, ax = plt.subplots(figsize=(5,4))
    for name, (fpr,tpr,aucv) in roc_dict.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={aucv:.3f})")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC – All Models")
    ax.legend(loc="lower right")
    st.pyplot(fig)

with st.sidebar:
    st.header("1) Data")
    up = st.file_uploader("Upload dataset (Excel/CSV) with 'Attrition' column", type=["xlsx","xls","csv"])

df = load_data(up)
if df is None or "Attrition" not in df.columns:
    st.stop()

y = make_binary_target(df["Attrition"]).astype(int)
X = df.drop(columns=["Attrition"])
preprocess, num_cols, cat_cols = build_preprocess(X)

st.sidebar.header("2) Global Filters")
jobrole_opts = sorted(df["JobRole"].dropna().unique().tolist()) if "JobRole" in df.columns else []
selected_roles = st.sidebar.multiselect("Filter: JobRole", options=jobrole_opts, default=jobrole_opts)

satisfaction_cols = [c for c in df.columns if c.lower().endswith("satisfaction")]
sat_col = st.sidebar.selectbox("Filter: Satisfaction column", options=satisfaction_cols if satisfaction_cols else [None])
sat_min = 1
if sat_col:
    try:
        lo, hi = int(df[sat_col].min()), int(df[sat_col].max())
        sat_min = st.sidebar.slider(f"Minimum {sat_col} (inclusive)", min_value=lo, max_value=hi, value=lo, step=1)
    except Exception:
        sat_col = None

def apply_all_filters():
    return apply_filters(df, selected_roles, sat_col, sat_min)

df_filt = apply_all_filters()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Insights", "📈 More Analytics", "🤖 Train & Evaluate Models", "🧪 Upload & Predict"])

with tab1:
    st.subheader("Insights – Actionable Cohorts")
    st.write(f"Records after filters: **{len(df_filt)}** / {len(df)}")

    if "JobRole" in df_filt.columns:
        jr = (df_filt.assign(Attrition_Bin=make_binary_target(df_filt["Attrition"]))
              .groupby("JobRole")["Attrition_Bin"]
              .agg(count="count", attrition_yes="sum"))
        jr["attrition_rate"] = jr["attrition_yes"]/jr["count"]
        jr = jr.reset_index().sort_values("attrition_rate", ascending=False)
        fig1 = px.bar(jr, x="JobRole", y="attrition_rate", hover_data=["count","attrition_yes"])
        fig1.update_layout(title="Attrition Rate by JobRole", xaxis_title="", yaxis_title="Attrition Rate")
        st.plotly_chart(fig1, use_container_width=True)

    if all(c in df_filt.columns for c in ["OverTime","BusinessTravel"]):
        tmp = df_filt.assign(Attrition_Bin=make_binary_target(df_filt["Attrition"]))
        pivot = (tmp.pivot_table(index="OverTime", columns="BusinessTravel", values="Attrition_Bin", aggfunc="mean"))
        fig2 = px.imshow(pivot, text_auto=True, aspect="auto")
        fig2.update_layout(title="Heatmap: Attrition by OverTime × BusinessTravel", xaxis_title="BusinessTravel", yaxis_title="OverTime")
        st.plotly_chart(fig2, use_container_width=True)

    if "DistanceFromHome" in df_filt.columns:
        tmp = df_filt.copy()
        tmp["Attrition_Bin"] = make_binary_target(tmp["Attrition"])
        try:
            tmp["dist_decile"] = pd.qcut(tmp["DistanceFromHome"], q=10, duplicates="drop")
            dist = tmp.groupby("dist_decile")["Attrition_Bin"].mean().reset_index()
            fig3 = px.line(dist, x="dist_decile", y="Attrition_Bin", markers=True)
            fig3.update_layout(title="Attrition vs DistanceFromHome (deciles)", xaxis_title="Distance decile", yaxis_title="Attrition Rate")
            st.plotly_chart(fig3, use_container_width=True)
        except Exception:
            pass

    if "MonthlyIncome" in df_filt.columns:
        tmp = df_filt.copy()
        tmp["Attrition_Bin"] = make_binary_target(tmp["Attrition"])
        tmp["Attrition_Label"] = tmp["Attrition_Bin"].map({0:"Stay",1:"Leave"})
        fig4 = px.box(tmp, x="Attrition_Label", y="MonthlyIncome", points="all")
        fig4.update_layout(title="Income Distribution by Attrition", xaxis_title="", yaxis_title="MonthlyIncome")
        st.plotly_chart(fig4, use_container_width=True)

    if "Age" in df_filt.columns:
        tmp = df_filt.copy()
        tmp["Attrition_Bin"] = make_binary_target(tmp["Attrition"])
        try:
            tmp["AgeBand"] = pd.cut(tmp["Age"], bins=[17,30,36,43,60], include_lowest=True)
            agg = tmp.groupby(["AgeBand","Attrition_Bin"]).size().reset_index(name="count")
            agg["Attrition_Label"] = agg["Attrition_Bin"].map({0:"Stay",1:"Leave"})
            fig5 = px.bar(agg, x="AgeBand", y="count", color="Attrition_Label", barmode="stack")
            fig5.update_layout(title="Counts by Age Band & Attrition", xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig5, use_container_width=True)
        except Exception:
            pass

with tab2:
    st.subheader("Deeper Analytics")
    st.write("Train quick models to get feature importances and combined ROC.")
    if st.button("Train quick models for analytics"):
        preprocess, _, _ = build_preprocess(X)
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
            "Gradient Boosted Trees": GradientBoostingClassifier(random_state=42, n_estimators=150, learning_rate=0.075, max_depth=3),
        }
        roc_dict = {}
        rows = []
        for name, mdl in models.items():
            pipe = Pipeline([("prep", preprocess), ("model", mdl)])
            metrics, cm, fpr, tpr, aucv = oof_eval(pipe, X, y)
            rows.append({"Model": name, **metrics})
            roc_dict[name] = (fpr, tpr, aucv)
            st.write(f"**{name} – Confusion Matrix (OOF)**")
            plot_bw_confusion(cm, title=f"{name} (OOF)")

        st.write("**Performance Table (5-fold CV)**")
        st.dataframe(pd.DataFrame(rows).set_index("Model"))

        st.write("**ROC Curves – All Models**")
        combined_roc_plot(roc_dict)

        st.write("**Top 15 Feature Importances – Gradient Boosted Trees**")
        gb_pipe = Pipeline([("prep", preprocess), ("model", GradientBoostingClassifier(random_state=42, n_estimators=150, learning_rate=0.075, max_depth=3))])
        gb_pipe.fit(X, y)
        cat_names = []
        cat_cols = [c for c in X.columns if c not in X.select_dtypes(include=[np.number]).columns]
        if len(cat_cols)>0:
            ohe = gb_pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
            cat_names = list(ohe.get_feature_names_out(cat_cols))
        all_names = X.select_dtypes(include=[np.number]).columns.tolist() + cat_names
        imps = gb_pipe.named_steps["model"].feature_importances_
        fi = (pd.DataFrame({"feature": all_names, "importance": imps})
              .sort_values("importance", ascending=False).head(15))
        st.dataframe(fi)
        fig, ax = plt.subplots(figsize=(6,5))
        ax.barh(fi["feature"][::-1], fi["importance"][::-1])
        ax.set_xlabel("Importance"); ax.set_ylabel("Feature"); ax.set_title("Top 15 – GBT")
        st.pyplot(fig)

with tab3:
    st.subheader("Train & Evaluate – Decision Tree / Random Forest / Gradient Boosted Trees")
    st.write("Click to compute Train/Test Accuracy, Precision, Recall, F1, ROC-AUC; plot confusion matrices and ROC.")
    if st.button("Run 5-fold CV"):
        preprocess, _, _ = build_preprocess(X)
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
            "Gradient Boosted Trees": GradientBoostingClassifier(random_state=42, n_estimators=150, learning_rate=0.075, max_depth=3),
        }
        roc_dict = {}
        rows = []
        for name, mdl in models.items():
            pipe = Pipeline([("prep", preprocess), ("model", mdl)])
            metrics, cm, fpr, tpr, aucv = oof_eval(pipe, X, y)
            rows.append({"Model": name, **metrics})
            roc_dict[name] = (fpr, tpr, aucv)

            st.markdown(f"**{name} – Confusion Matrix (OOF)**")
            plot_bw_confusion(cm, title=f"{name} (OOF)")

            fig, ax = plt.subplots(figsize=(5,4))
            ax.plot(fpr, tpr, label=f"AUC={aucv:.3f}")
            ax.plot([0,1],[0,1], linestyle="--")
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC – {name}")
            ax.legend(loc="lower right")
            st.pyplot(fig)

        st.write("**Performance Table (5-fold CV)**")
        metrics_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(metrics_df)

        st.write("**Combined ROC**")
        combined_roc_plot(roc_dict)

with tab4:
    st.subheader("Upload New Dataset & Predict Attrition")
    st.write("Train a model on current data, then upload a new dataset (same schema) to score and download predictions.")
    model_choice = st.selectbox("Choose model", ["Gradient Boosted Trees","Random Forest","Decision Tree"])
    if st.button("Train model"):
        preprocess, _, _ = build_preprocess(X)
        if model_choice == "Gradient Boosted Trees":
            mdl = GradientBoostingClassifier(random_state=42, n_estimators=150, learning_rate=0.075, max_depth=3)
        elif model_choice == "Random Forest":
            mdl = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced_subsample")
        else:
            mdl = DecisionTreeClassifier(random_state=42, class_weight="balanced")
        model_pipe = Pipeline([("prep", preprocess), ("model", mdl)])
        model_pipe.fit(X, y)
        st.success("Model trained on current dataset.")
        st.session_state["trained_model"] = model_pipe

    up2 = st.file_uploader("Upload new dataset to predict (Excel/CSV)", type=["xlsx","xls","csv"], key="predict_uploader")
    if up2 is not None and "trained_model" in st.session_state:
        def load_any(file):
            name = file.name.lower()
            if name.endswith(".xlsx") or name.endswith(".xls"):
                return pd.read_excel(file)
            return pd.read_csv(file)
        newdf = load_any(up2)
        newdf.columns = [str(c).strip() for c in newdf.columns]
        newdf = newdf.dropna(axis=1, how="all").dropna(axis=0, how="all")

        model_pipe = st.session_state["trained_model"]
        newX = newdf.drop(columns=[c for c in newdf.columns if c.lower()=="attrition"], errors="ignore")
        try:
            proba = model_pipe.predict_proba(newX)[:,1]
        except Exception:
            dec = model_pipe.decision_function(newX)
            dec_min, dec_max = dec.min(), dec.max()
            proba = (dec - dec_min)/(dec_max-dec_min) if dec_max>dec_min else np.zeros_like(dec, dtype=float)
        preds = (proba >= 0.5).astype(int)
        out = newdf.copy()
        out["Attrition_Pred"] = preds
        out["Attrition_Pred_Prob"] = np.round(proba, 4)

        st.write("Preview of predictions:")
        st.dataframe(out.head(20))
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions_with_attrition.csv", mime="text/csv")
    elif up2 is not None and "trained_model" not in st.session_state:
        st.info("Please train a model first (click 'Train model').")

st.markdown("---")
st.caption("Use the left sidebar to filter Job Roles and a satisfaction column. All charts in the Insights tab honor these filters.")
