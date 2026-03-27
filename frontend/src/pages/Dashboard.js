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

import API_BASE from "../config";   // ✅ USE ENV
import "./Dashboard.css";

function Dashboard() {

  const navigate = useNavigate();

  const [kpis, setKpis] = useState(null);
  const [forecast, setForecast] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [blockchain, setBlockchain] = useState("");
  const [riskData, setRiskData] = useState([]);
  const [comparisonData, setComparisonData] = useState([]);
  const [loading, setLoading] = useState(true);

  const COLORS = ["#22c55e", "#f59e0b", "#ef4444"];

  /* ================================
     AUTH CHECK + AUTO REFRESH
  ================================== */

  useEffect(() => {

    const token = localStorage.getItem("token");

    if (!token) {
      navigate("/");
      return;
    }

    loadDashboard(token);

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

      const res = await axios.get(
        `${API_BASE}/dashboard-data`,   // ✅ FIXED
        {
          headers: {
            Authorization: `Bearer ${token}`
          }
        }
      );

      const data = res.data;

      setKpis(data.kpis || {});
      setForecast(data.forecast || []);
      setPrediction(data.prediction || {});
      setComparisonData(data.chart || []);

      /* ===== ✅ FIXED RISK DISTRIBUTION ===== */

      const anomaly = data.anomaly || {};

      setRiskData([
        { name: "Low", value: anomaly.low || 0 },
        { name: "Medium", value: anomaly.medium || 0 },
        { name: "High", value: anomaly.high || 0 }
      ]);

      setBlockchain(data.blockchain?.status || "Unknown");

    } catch (error) {

      console.error("Dashboard Load Error:", error);

    } finally {

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
        `${API_BASE}/upload-csv`,   // ✅ FIXED
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`
          }
        }
      );

      alert("CSV uploaded successfully ✅");

      loadDashboard(token);

    } catch (error) {

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
     LOADING
  ================================== */

  if (loading) {
    return (
      <h2 style={{ textAlign: "center", marginTop: "120px" }}>
        Loading Dashboard...
      </h2>
    );
  }

  /* ================================
     FORECAST DATA
  ================================== */

  const forecastChart = [...forecast];

  if (prediction?.next_month_prediction !== undefined) {
    forecastChart.push({
      month: "Next",
      revenue: prediction.next_month_prediction
    });
  }

  const totalRecords = riskData.reduce((a, b) => a + b.value, 0);

  /* ================================
     UI
  ================================== */

  return (

    <div className="dashboard">

      {/* SIDEBAR */}
      <div className="sidebar">

        <h2>FinPulse</h2>

        <button onClick={logout}>Logout</button>

        <div className="security-card">
          <h3>Records Analysed</h3>
          <p>{totalRecords}</p>
        </div>

        <div className="security-card">
          <h3>Blockchain</h3>
          <p>{blockchain}</p>
        </div>

      </div>

      {/* MAIN */}
      <div className="main">

        <h1>Financial Dashboard</h1>

        {/* Upload */}
        <div className="upload-box">
          <input type="file" accept=".csv" onChange={uploadCSV} />
        </div>

        {/* KPI */}
        <div className="kpi-container">

          <div className="kpi-card">
            <h3>Total Revenue</h3>
            <p>₹ {Number(kpis.total_revenue || 0).toLocaleString()}</p>
          </div>

          <div className="kpi-card">
            <h3>Total Expense</h3>
            <p>₹ {Number(kpis.total_expense || 0).toLocaleString()}</p>
          </div>

          <div className="kpi-card">
            <h3>Net Profit</h3>
            <p>₹ {Number(kpis.net_profit || 0).toLocaleString()}</p>
          </div>

        </div>

        {/* CHARTS */}
        <div className="chart-row">

          {/* LINE */}
          <div className="chart-card">

            <h3>Revenue Forecast</h3>

            <LineChart width={500} height={300} data={forecastChart}>
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <CartesianGrid />
              <Line type="monotone" dataKey="revenue" stroke="#38bdf8" />
            </LineChart>

          </div>

          {/* PIE */}
          <div className="chart-card">

            <h3>Risk Distribution</h3>

            <PieChart width={350} height={300}>
              <Pie data={riskData} dataKey="value" outerRadius={100} label>
                {riskData.map((entry, index) => (
                  <Cell key={index} fill={COLORS[index]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>

          </div>

        </div>

        {/* BAR */}
        <div className="chart-card">

          <h3>Revenue vs Expense</h3>

          <BarChart width={700} height={300} data={comparisonData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="revenue" fill="#6366f1" />
            <Bar dataKey="expense" fill="#22c55e" />
          </BarChart>

        </div>

      </div>

    </div>

  );
}

export default Dashboard;