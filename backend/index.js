const express = require('express');
const bodyParser = require('body-parser')
const app = express();
const WebSocket = require('ws');
const PORT = process.env.PORT || process.env.port || 3000;
const server = new WebSocket.Server({ server: app.listen(PORT) });

server.on('connection', (socket) => {
  socket.send(JSON.stringify({ startingScore: 100 }));
});

app.use(bodyParser.json())

app.post('/get-prediction-data', (req, res) => {
  try {
    
    if (req.body) {
      server.clients.forEach(client => {
        client.send(JSON.stringify(req.body));
      });
    }
    return res.status(200).json({
      success: true
    }).end();
  } catch (err) {
    console.log(err);
  }
  return res.status(400).json({
    success: false
  }).end();
});
