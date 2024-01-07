# Visualization Toolkit

The Visualization Toolkit is a simple HTML/Javascript Application, that works best in Firefox.
It accesses a local server on port 5111 that serves the learning data in the json format, implementations of deep learning frameworks may implement the bellow mentioned HTML protocol to allow a connection and visualization of deep learning applications.

## Protocol
- `GET http://localhost:5111/<id>`
  the server either returns all epoch and batch informations that have been collected since the learning process has been started if the client id is not registered yet (and registers the id) or all epoch and batch informations that have been collected since the last connection.
  The data is given in the following json format: `{state: <"play" | "pause">, batches: [{batch: <batch number>, error: <batch error>}], epochs: [{epoch: <epoch number>, error: <mean epoch error>, validation_error: <validation_error>}], total_batches: <number of batches>}`
- `GET http://localhost:5111/pause`
  the trainings process in paused before the next batch is started (it is not stopped, it merely blocks until a `start` request is received)
- `GET http://localhost:5111/start`
  the trainings process is continued if it has been paused previously
- `GET http://localhost:5111/stop`
  the trainings process is stopped before the next batch would be started. All code that should be run after the trainings process should be run.
- `GET http://localhost:5111/descripe`
  A description of the current model is returned in the following format: `{layers: [{name: <layer name>, description: <layer description>, no_params: <number of parameters}], loss_fct: <name of loss function>, optimizer: <name of optimizer>}`
