import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  Legend
} from "recharts";

import "./Dashboard.css";

function Dashboard() {

  const navigate = useNavigate();

  const [kpis, setKpis] = useState(null);
  const [forecast, setForecast] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [risk, setRisk] = useState("");
  const [blockchain, setBlockchain] = useState("");
  const [riskData, setRiskData] = useState([]);
  const [comparisonData, setComparisonData] = useState([]);
  const [loading, setLoading] = useState(true);

  const COLORS = ["#22c55e", "#f59e0b", "#ef4444"];

  /* ================================
     AUTH CHECK + REAL TIME REFRESH
  ================================== */

  useEffect(() => {

    const token = localStorage.getItem("token");

    if (!token) {
      navigate("/");
      return;
    }

    loadDashboard(token);

    // 🔴 Real-time monitoring (refresh every 10 sec)
    const interval = setInterval(() => {
      loadDashboard(token, false);
    }, 10000);

    return () => clearInterval(interval);

  }, [navigate]);

  /* ================================
     LOAD DASHBOARD DATA
  ================================== */

  const loadDashboard = async (token, showLoader = true) => {

    try {

      if (showLoader) setLoading(true);

      const headers = {
        Authorization: `Bearer ${token}`
      };

      const res = await axios.get(
        "http://127.0.0.1:8000/dashboard-data",
        { headers }
      );

      const data = res.data;

      setKpis(data.kpis);
      setForecast(data.forecast);
      setPrediction(data.prediction);
      setComparisonData(data.chart);

      /* ===== Risk Distribution ===== */

      const results = data.anomaly?.results || [];

      let low = 0;
      let medium = 0;
      let high = 0;

      results.forEach(item => {

        if (item.classified_risk === "Low") low++;
        else if (item.classified_risk === "Medium") medium++;
        else if (item.classified_risk === "High") high++;

      });

      setRiskData([
        { name: "Low Risk", value: low },
        { name: "Medium Risk", value: medium },
        { name: "High Risk", value: high }
      ]);

      setRisk(`Total Records Analysed: ${data.risk_records?.total_records || 0}`);
      setBlockchain(data.blockchain?.status || "Unknown");

    }
    catch (error) {

      console.error("Dashboard Load Error:", error);

    }
    finally {

      if (showLoader) setLoading(false);

    }

  };

  /* ================================
     CSV UPLOAD
  ================================== */

  const uploadCSV = async (e) => {

    const file = e.target.files[0];
    if (!file) return;

    const token = localStorage.getItem("token");

    const formData = new FormData();
    formData.append("file", file);

    try {

      await axios.post(
        "http://127.0.0.1:8000/upload-csv",
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`
          }
        }
      );

      alert("CSV uploaded successfully ✅");

      loadDashboard(token);

    }
    catch (error) {

      console.error("Upload error:", error);
      alert("CSV upload failed ❌");

    }

  };

  /* ================================
     LOGOUT
  ================================== */

  const logout = () => {

    localStorage.removeItem("token");
    navigate("/");

  };

  /* ================================
     LOADING SCREEN
  ================================== */

  if (loading) {

    return (
      <h2 style={{ textAlign: "center", marginTop: "120px" }}>
        Loading Financial Analytics Dashboard...
      </h2>
    );

  }

  /* ================================
     FORECAST DATA
  ================================== */

  const forecastChart = [...forecast];

  if (prediction) {

    forecastChart.push({
      month: "Next",
      revenue: prediction.next_month_prediction
    });

  }

  /* ================================
     DASHBOARD UI
  ================================== */

  return (

    <div className="dashboard">

      {/* SIDEBAR */}

      <div className="sidebar">

        <h2>FinPulse</h2>

        <button onClick={logout}>
          Logout
        </button>

        <div className="sidebar-security">

          <div className="security-card">
            <h3>Records Analysed</h3>
            <p>{risk}</p>
          </div>

          <div className="security-card">
            <h3>Blockchain Integrity</h3>
            <p>{blockchain}</p>
          </div>

        </div>

      </div>

      {/* MAIN AREA */}

      <div className="main">

        <h1>Financial Analytics Dashboard</h1>

        {/* CSV Upload */}

        <div className="upload-box">

          <h3>Upload Financial CSV</h3>

          <input
            type="file"
            accept=".csv"
            onChange={uploadCSV}
          />

        </div>

        {/* KPI CARDS */}

        <div className="kpi-container">

          <div className="kpi-card">
            <h3>Total Revenue</h3>
            <p>₹ {Number(kpis?.total_revenue || 0).toLocaleString()}</p>
          </div>

          <div className="kpi-card">
            <h3>Total Expense</h3>
            <p>₹ {Number(kpis?.total_expense || 0).toLocaleString()}</p>
          </div>

          <div className="kpi-card">
            <h3>Net Profit</h3>
            <p>₹ {Number(kpis?.net_profit || 0).toLocaleString()}</p>
          </div>

        </div>

        {/* CHARTS */}

        <div className="chart-row">

          {/* FORECAST */}

          <div className="chart-card">

            <h3>Revenue Forecast</h3>

            <LineChart width={500} height={300} data={forecastChart}>

              <XAxis dataKey="month" stroke="#94a3b8"/>

              <YAxis
                width={90}
                tickFormatter={(value) => value.toLocaleString()}
                stroke="#94a3b8"
              />

              <Tooltip formatter={(value) => value.toLocaleString()}/>

              <CartesianGrid stroke="#334155"/>

              <Line
                type="monotone"
                dataKey="revenue"
                stroke="#38bdf8"
                strokeWidth={3}
                dot={{ r: 5 }}
                activeDot={{ r: 9 }}
              />

            </LineChart>

          </div>

          {/* RISK PIE */}

          <div className="chart-card">

            <h3>Risk Distribution</h3>

            <PieChart width={350} height={300}>

              <Pie
                data={riskData}
                dataKey="value"
                cx="50%"
                cy="50%"
                outerRadius={100}
                label
              >

                {riskData.map((entry,index)=>(
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}

              </Pie>

              <Tooltip/>

            </PieChart>

          </div>

        </div>

        {/* BAR CHART */}

        <div className="chart-card">

          <h3>Revenue vs Expense</h3>

          <BarChart width={700} height={300} data={comparisonData}>

            <CartesianGrid strokeDasharray="3 3" stroke="#334155"/>

            <XAxis dataKey="month" stroke="#94a3b8"/>

            <YAxis
              width={90}
              tickFormatter={(value) => value.toLocaleString()}
              stroke="#94a3b8"
            />

            <Tooltip formatter={(value) => value.toLocaleString()}/>

            <Legend/>

            <Bar dataKey="revenue" fill="#6366f1"/>
            <Bar dataKey="expense" fill="#22c55e"/>

          </BarChart>

        </div>

      </div>

    </div>

  );

}

export default Dashboard;