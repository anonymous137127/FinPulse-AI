import { useState } from "react";
import axios from "axios";
import { useNavigate, Link } from "react-router-dom";
import "./Login.css";

// Backend API
import API_BASE from "../config";

function Register() {

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();

  const handleRegister = async (e) => {

    e.preventDefault();

    const cleanUsername = username.trim();
    const cleanPassword = password.trim();

    if (!cleanUsername || !cleanPassword || !role) {
      setMessage("⚠️ Please enter username, password and role");
      return;
    }

    try {

      setLoading(true);
      setMessage("");

      await axios.post(
        `${API_BASE}/register`,
        null,
        {
          params: {
            username: cleanUsername,
            password: cleanPassword,
            role: role
          }
        }
      );

      setMessage("✅ Registration successful");

      // Redirect to login
      setTimeout(() => {
        navigate("/");
      }, 1000);

    } catch (error) {

      console.error("Register Error:", error);

      if (error.response) {
        setMessage(error.response.data.detail || "❌ Registration failed");
      } else {
        setMessage("❌ Server not reachable");
      }

    } finally {

      setLoading(false);

    }

  };

  return (

    <div className="login-container">

      <div className="login-card">

        <h2 className="title">FinPulse</h2>
        <p className="subtitle">Create your account</p>

        {message && (
          <div className="popup">
            {message}
          </div>
        )}

        <form onSubmit={handleRegister}>

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

          <select
            className="role-select"
            value={role}
            onChange={(e) => setRole(e.target.value)}
          >
            <option value="">Select Role</option>
            <option value="analyst">Analyst</option>
            <option value="auditor">Auditor</option>
            <option value="admin">Admin</option>
          </select>

          <button
            type="submit"
            className="login-btn"
            disabled={loading}
          >
            {loading ? "Registering..." : "Register"}
          </button>

        </form>

        <p className="register-text">
          Already have an account? <Link to="/">Login</Link>
        </p>

      </div>

    </div>

  );
}

export default Register;