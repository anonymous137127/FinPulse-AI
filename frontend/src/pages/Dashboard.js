import { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
  PieChart, Pie, Cell,
  BarChart, Bar, Legend,
  ResponsiveContainer
} from "recharts";

import API_BASE from "../config";
import "./Dashboard.css";

function Dashboard() {
  const navigate = useNavigate();

  const [kpis, setKpis]                     = useState(null);
  const [forecast, setForecast]             = useState([]);
  const [prediction, setPrediction]         = useState(null);
  const [blockchain, setBlockchain]         = useState("");
  const [riskData, setRiskData]             = useState([]);
  const [comparisonData, setComparisonData] = useState([]);
  const [loading, setLoading]               = useState(true);
  const [sidebarOpen, setSidebarOpen]       = useState(false);

  const sidebarRef = useRef(null);
  const COLORS     = ["#22c55e", "#f59e0b", "#ef4444"];

  /* close sidebar on outside click */
  useEffect(() => {
    const fn = (e) => {
      if (sidebarOpen && sidebarRef.current &&
          !sidebarRef.current.contains(e.target) &&
          !e.target.closest(".hamburger")) setSidebarOpen(false);
    };
    document.addEventListener("mousedown", fn);
    return () => document.removeEventListener("mousedown", fn);
  }, [sidebarOpen]);

  /* close on desktop resize */
  useEffect(() => {
    const fn = () => { if (window.innerWidth > 768) setSidebarOpen(false); };
    window.addEventListener("resize", fn);
    return () => window.removeEventListener("resize", fn);
  }, []);

  /* lock scroll */
  useEffect(() => {
    document.body.style.overflow = sidebarOpen ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [sidebarOpen]);

  /* auth + auto-refresh */
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) { navigate("/"); return; }
    loadDashboard(token);
    const id = setInterval(() => loadDashboard(token, false), 10000);
    return () => clearInterval(id);
  }, [navigate]);

  const loadDashboard = async (token, showLoader = true) => {
    try {
      if (showLoader) setLoading(true);
      const { data } = await axios.get(`${API_BASE}/dashboard-data`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setKpis(data.kpis || {});
      setForecast(data.forecast || []);
      setPrediction(data.prediction || {});
      setComparisonData(data.chart || []);
      const a = data.anomaly || {};
      setRiskData([
        { name: "Low",    value: a.low    || 0 },
        { name: "Medium", value: a.medium || 0 },
        { name: "High",   value: a.high   || 0 },
      ]);
      setBlockchain(data.blockchain?.status || "Unknown");
    } catch (err) {
      console.error("Dashboard Load Error:", err);
    } finally {
      if (showLoader) setLoading(false);
    }
  };

  const uploadCSV = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const token = localStorage.getItem("token");
    const fd    = new FormData();
    fd.append("file", file);
    try {
      await axios.post(`${API_BASE}/upload-csv`, fd, {
        headers: { Authorization: `Bearer ${token}` },
      });
      alert("CSV uploaded successfully ✅");
      loadDashboard(token);
    } catch {
      alert("CSV upload failed ❌");
    }
  };

  const logout = () => { localStorage.removeItem("token"); navigate("/"); };

  if (loading)
    return <h2 style={{ textAlign: "center", marginTop: 120, color: "#38bdf8" }}>Loading Dashboard...</h2>;

  const forecastChart = [...forecast];
  if (prediction?.next_month_prediction !== undefined)
    forecastChart.push({ month: "Next", revenue: prediction.next_month_prediction });

  const totalRecords = riskData.reduce((a, b) => a + b.value, 0);

  const tooltipProps = {
    contentStyle: { background: "#020617", border: "1px solid #38bdf8", borderRadius: 10, fontSize: 13 },
    labelStyle:   { color: "#e2e8f0" },
    itemStyle:    { color: "#22c55e", fontWeight: "bold" },
  };

  return (
    <div className="dashboard">

      {/* HAMBURGER */}
      <button
        className={`hamburger${sidebarOpen ? " active" : ""}`}
        onClick={() => setSidebarOpen((p) => !p)}
        aria-label="Toggle menu"
      >
        <span /><span /><span />
      </button>

      {/* OVERLAY */}
      <div className={`overlay${sidebarOpen ? " active" : ""}`} onClick={() => setSidebarOpen(false)} />

      {/* SIDEBAR */}
      <div className={`sidebar${sidebarOpen ? " open" : ""}`} ref={sidebarRef}>
        <h2>FinPulse</h2>
        <button onClick={logout}>Logout</button>
        <div className="sidebar-security">
          <div className="security-card">
            <h3>Records Analysed</h3>
            <p>{totalRecords}</p>
          </div>
          <div className="security-card">
            <h3>Blockchain</h3>
            <p>{blockchain}</p>
          </div>
        </div>
      </div>

      {/* MAIN */}
      <div className="main">
        <h1>Financial Dashboard</h1>

        {/* Upload */}
        <div className="upload-box">
          <input type="file" accept=".csv" onChange={uploadCSV} />
        </div>

        {/* KPIs */}
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

        {/* TOP ROW: Line (60%) + Pie (40%) */}
        <div className="chart-row">

          <div className="chart-card chart-card--line">
            <h3>Revenue Forecast</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={forecastChart} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="month" tick={{ fill: "#94a3b8", fontSize: 12 }} axisLine={{ stroke: "#1e293b" }} tickLine={false} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={{ stroke: "#1e293b" }} tickLine={false} width={65} />
                <Tooltip {...tooltipProps} />
                <Line
                  type="monotone" dataKey="revenue" stroke="#38bdf8" strokeWidth={2.5}
                  dot={{ fill: "#020617", stroke: "#38bdf8", strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6, fill: "#38bdf8" }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card chart-card--pie">
            <h3>Risk Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart margin={{ top: 10, right: 30, left: 30, bottom: 10 }}>
                <Pie
                  data={riskData} dataKey="value"
                  cx="50%" cy="50%"
                  outerRadius="60%" innerRadius="28%"
                  paddingAngle={3} label={false}
                >
                  {riskData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                </Pie>
                <Tooltip {...tooltipProps} />
                <Legend iconType="circle" iconSize={10} wrapperStyle={{ color: "#94a3b8", fontSize: 13, paddingTop: 10 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>

        </div>

        {/* BAR — full width */}
        <div className="chart-card chart-card--bar">
          <h3>Revenue vs Expense</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparisonData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="month" tick={{ fill: "#94a3b8", fontSize: 12 }} axisLine={{ stroke: "#1e293b" }} tickLine={false} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={{ stroke: "#1e293b" }} tickLine={false} width={65} />
              <Tooltip {...tooltipProps} />
              <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 13 }} />
              <Bar dataKey="revenue" fill="#6366f1" radius={[4, 4, 0, 0]} maxBarSize={40} />
              <Bar dataKey="expense"  fill="#22c55e" radius={[4, 4, 0, 0]} maxBarSize={40} />
            </BarChart>
          </ResponsiveContainer>
        </div>

      </div>
    </div>
  );
}

export default Dashboard;