import React, { useEffect, useState } from 'react';
import mqtt from 'mqtt';

const BROKER_URL = import.meta.env.VITE_BROKER_URL || 'wss://broker.hivemq.com:8884/mqtt';
const STATE_TOPIC = import.meta.env.VITE_STATE_TOPIC || 'demo/solenoid/state';
const COMMAND_TOPIC = import.meta.env.VITE_COMMAND_TOPIC || 'demo/solenoid/command';

export default function SolenoidControl() {
  const [client, setClient] = useState(null);
  const [status, setStatus] = useState('disconnected');
  const [solenoidState, setSolenoidState] = useState('unknown');

  useEffect(() => {
    const mqttClient = mqtt.connect(BROKER_URL);

    mqttClient.on('connect', () => {
      setStatus('connected');
      mqttClient.subscribe(STATE_TOPIC);
    });

    mqttClient.on('reconnect', () => setStatus('reconnecting'));
    mqttClient.on('error', () => setStatus('error'));

    mqttClient.on('message', (topic, message) => {
      if (topic === STATE_TOPIC) {
        setSolenoidState(message.toString());
      }
    });

    setClient(mqttClient);

    return () => {
      mqttClient.end();
    };
  }, []);

  const toggleSolenoid = () => {
    if (client && status === 'connected') {
      const nextState = solenoidState === 'ON' ? 'OFF' : 'ON';
      client.publish(COMMAND_TOPIC, nextState);
    }
  };

  return (
    <div>
      <p>Connection status: {status}</p>
      <p className="status">Solenoid state: {solenoidState}</p>
      <button onClick={toggleSolenoid} disabled={status !== 'connected'}>
        Toggle Solenoid
      </button>
    </div>
  );
}
