import React, { useState } from "react";
import axios from "axios";
import { TextField, Button, Box, Typography, Paper, Checkbox, FormControlLabel, CircularProgress } from "@mui/material";
import { FormControl, FormLabel, RadioGroup, Radio } from "@mui/material";
import { Tooltip } from "@mui/material";

import { PieChart, Pie, Cell, Legend, ResponsiveContainer } from "recharts";


function App() {
  const [text, setText] = useState("");
  const [response, setResponse] = useState(null);
  const [selectedModel, setSelectedModel] = useState("XGBC"); // Default model
  const [includeShap, setIncludeShap] = useState(false); // Checkbox state
  const [loading, setLoading] = useState(false);


  const handleSubmit = async () => {
    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", {
          text,
          explain: includeShap,
          classifier: selectedModel,// Send checkbox state to backend
           });
      setResponse(res.data);
    } catch (error) {
      console.error("Very bad error:", error);
    }
    finally {
      setLoading(false);
    }
  };

  const COLORS = ["#63B7AF", "#FF7F50"];

  return (
    <Box sx={{ p: 3, minWidth: 650, maxWidth: 650, mx: "auto", mt: 5 }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
        <Typography variant="h4" gutterBottom align="center">
          AI detector
        </Typography>

        {/* Text Input */}
        <TextField
          fullWidth
          multiline
          rows={4}
          label="Enter your text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          sx={{
            mb: 3,
            "& .MuiOutlinedInput-root": {
              borderRadius: 2,
            },
          }}
        />

          {/* Checkbox to enable SHAP explanations */}
        <FormControlLabel
          control={<Checkbox checked={includeShap} onChange={(e) => setIncludeShap(e.target.checked)} />}
          label="Include SHAP explanations"
          sx={{ mb: 2 }}
        />

      <FormControl component="fieldset" sx={{ mb: 2 }}>
        <FormLabel component="legend">Select Classifier</FormLabel>
              <RadioGroup
                    row
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                >
                <Tooltip title="Gradient Boosted Trees model – fast and interpretable" arrow>
                    <FormControlLabel value="XGBC" control={<Radio />} label="XGBC" />
                </Tooltip>

                <Tooltip title="Transformer-based language model – deep and powerful" arrow>
                    <FormControlLabel value="BERT" control={<Radio />} label="BERT" />
                </Tooltip>

                <Tooltip title="Combined model - best of both worlds." arrow>
                    <FormControlLabel value="Combined" control={<Radio />} label="Combined" />
                </Tooltip>

              </RadioGroup>
        </FormControl>

        {/* Submit Button */}
        <Box sx={{ display: "flex", justifyContent: "center", mt: 2 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleSubmit}
            disabled={loading}
            sx={{
              px: 4,
              py: 1,
              borderRadius: 2,
              textTransform: "none",
              fontWeight: "bold",
              minWidth: 120,
            }}
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : "Submit"}
          </Button>
        </Box>



        {/* Response Display */}
        {response && (
          <Box sx={{ mt: 4 }}>
            {/* Prediction Chart */}
            <Typography variant="h6" gutterBottom align="center">
              Prediction Chart
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={response?.prediction?.[0] || []}
                  dataKey="score"
                  nameKey="label"
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  fill="#8884d8"
                  label={(entry) => `${entry.label}: ${(entry.score * 100).toFixed(1)}%`}
                >
                  {response.prediction.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Legend />
              </PieChart>
            </ResponsiveContainer>

            {includeShap && response.shap_explanation && (
                    <Box sx={{ mt: 3 }}>
                    <Typography variant="h6" gutterBottom align="center">
                            SHAP Explanation
                    </Typography>
                        <iframe
                            src={`http://127.0.0.1:8000${response.shap_explanation}`} // Full URL to access the SHAP file
                            width="100%"
                            height="600px"
                            title="SHAP Explanation"
                            style={{ border: "none" }}
                            />
                     </Box>
            )}
          </Box>
        )}
      </Paper>
    </Box>
  );
}

export default App;
