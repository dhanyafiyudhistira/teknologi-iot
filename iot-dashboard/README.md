# IoT Solenoid Dashboard

This is a simple React dashboard created with Vite to remotely control a solenoid valve using MQTT over WebSockets. The UI is lightweight and mobile friendly, built with React hooks.

## Available Scripts

```bash
npm install
npm run dev       # start development server
npm run build     # build for production
npm run preview   # preview production build
```

## MQTT Setup

The example uses the public HiveMQ broker via WebSockets:

```
wss://broker.hivemq.com:8884/mqtt
```

Topics:
- `demo/solenoid/state` – publishes the current state (`ON` or `OFF`).
- `demo/solenoid/command` – subscribe to this topic to receive toggle commands.

You can adapt `src/components/SolenoidControl.jsx` to point to your own broker (e.g., Mosquitto or EMQX) by updating the `BROKER_URL`, `STATE_TOPIC`, and `COMMAND_TOPIC` constants.

The dashboard subscribes to the state topic and publishes to the command topic when the toggle button is clicked.
