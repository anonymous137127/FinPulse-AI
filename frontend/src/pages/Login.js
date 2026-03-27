import { useState } from "react";
import axios from "axios";
import { useNavigate, Link } from "react-router-dom";
import API_BASE from "../config";   // ✅ USE ENV
import "./Login.css";

function Login() {

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");

  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();

    if (!username || !password) {
      setMessage("⚠️ Please enter username and password");
      return;
    }

    try {

      const response = await axios.post(
        `${API_BASE}/login`,   // ✅ FIXED (NO LOCALHOST)
        null,
        {
          params: {
            username,
            password
          }
        }
      );

      // ✅ Save token
      localStorage.setItem("token", response.data.access_token);

      setMessage("✅ Login Successful");

      setTimeout(() => {
        navigate("/dashboard");
      }, 1000);

    } catch (error) {

      console.error(error);

      if (error.response) {
        setMessage(error.response.data.detail || "❌ Login failed");
      } else {
        setMessage("❌ Server not reachable");
      }

    }
  };

  return (

    <div className="login-container">

      <div className="login-card">

        <h2 className="title">FinPulse</h2>
        <p className="subtitle">AI Financial Intelligence Platform</p>

        <form onSubmit={handleLogin}>

          <input
            type="text"
            placeholder="Username"
            className="input"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />

          <input
            type="password"
            placeholder="Password"
            className="input"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          <button type="submit" className="login-btn">
            Login
          </button>

        </form>

        {message && <div className="popup">{message}</div>}

        <p className="register-text">
          Don't have an account? <Link to="/register">Register</Link>
        </p>

      </div>

      <div className="security-text">
        <div className="scroll-text">
          🔐 Secure Financial Analytics System •
          ⚡ AI-Powered Revenue Forecasting •
          📊 Risk Detection Engine •
          ⛓ Blockchain Verification •
        </div>
      </div>

    </div>

  );

}

export default Login;