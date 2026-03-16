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
import API_BASE from "../config";

function Dashboard() {

  const navigate = useNavigate();

  const [kpis, setKpis] = useState({});
  const [forecast, setForecast] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [risk, setRisk] = useState("");
  const [blockchain, setBlockchain] = useState("Unknown");
  const [riskData, setRiskData] = useState([]);
  const [comparisonData, setComparisonData] = useState([]);
  const [loading, setLoading] = useState(true);

  const COLORS = ["#22c55e", "#f59e0b", "#ef4444"];

  const formatCurrency = (num) => {
    return `₹ ${Number(num || 0).toLocaleString()}`;
  };

  useEffect(() => {

    const token = localStorage.getItem("token");

    if (!token) {
      navigate("/");
      return;
    }

    loadDashboard(token);

    const interval = setInterval(() => {
      loadDashboard(token, false);
    }, 30000);

    return () => clearInterval(interval);

  }, [navigate]);

  const loadDashboard = async (token, showLoader = true) => {

    try {

      if (showLoader) setLoading(true);

      const res = await axios.get(
        `${API_BASE}/dashboard-data`,
        {
          headers: {
            Authorization: `Bearer ${token}`
          }
        }
      );

      const data = res.data;

      setKpis(data?.kpis || {});
      setForecast(data?.forecast || []);
      setPrediction(data?.prediction || null);
      setComparisonData(data?.chart || []);

      const results = data?.anomaly?.results ?? [];

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

      setRisk(`Total Records Analysed: ${data?.risk_records?.total_records || 0}`);

      setBlockchain(data?.blockchain?.status || "Unknown");

    }
    catch (error) {

      console.error("Dashboard Load Error:", error);

      if (error.response?.status === 401) {
        localStorage.removeItem("token");
        navigate("/");
      }

    }
    finally {

      if (showLoader) setLoading(false);

    }

  };

  const uploadCSV = async (e) => {

    const file = e.target.files[0];
    if (!file) return;

    const token = localStorage.getItem("token");

    const formData = new FormData();
    formData.append("file", file);

    try {

      await axios.post(
        `${API_BASE}/upload-csv`,
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "multipart/form-data"
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

  const logout = () => {

    localStorage.removeItem("token");
    navigate("/");

  };

  if (loading) {

    return (
      <h2 style={{ textAlign: "center", marginTop: "120px" }}>
        Loading Financial Analytics Dashboard...
      </h2>
    );

  }

  const forecastChart = [...forecast];

  if (prediction && prediction.next_month_prediction !== undefined) {

    forecastChart.push({
      month: "Next",
      revenue: prediction.next_month_prediction
    });

  }

  return (

    <div className="dashboard">

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
            <p style={{ color: blockchain === "Valid" ? "#22c55e" : "#ef4444" }}>
              {blockchain}
            </p>
          </div>

        </div>

      </div>

      <div className="main">

        <h1>Financial Analytics Dashboard</h1>

        <div className="upload-box">

          <h3>Upload Financial CSV</h3>

          <input
            type="file"
            accept=".csv"
            onChange={uploadCSV}
          />

        </div>

        <div className="kpi-container">

          <div className="kpi-card">
            <h3>Total Revenue</h3>
            <p>{formatCurrency(kpis?.total_revenue)}</p>
          </div>

          <div className="kpi-card">
            <h3>Total Expense</h3>
            <p>{formatCurrency(kpis?.total_expense)}</p>
          </div>

          <div className="kpi-card">
            <h3>Net Profit</h3>
            <p>{formatCurrency(kpis?.net_profit)}</p>
          </div>

        </div>

        <div className="chart-row">

          <div className="chart-card">

            <h3>Revenue Forecast</h3>

            <LineChart width={500} height={300} data={forecastChart}>
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <CartesianGrid stroke="#ccc" />
              <Line
                type="monotone"
                dataKey="revenue"
                stroke="#3b82f6"
                strokeWidth={3}
              />
            </LineChart>

          </div>

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

                {riskData.map((entry, index) => (
                  <Cell
                    key={index}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}

              </Pie>

              <Tooltip />

            </PieChart>

          </div>

        </div>

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