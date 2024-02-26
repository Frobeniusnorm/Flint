function sleep(milliseconds) {
	return new Promise(resolve => setTimeout(resolve, milliseconds));
}

const id = new Date().getTime();

let gd_epochs = [];
let gd_batches = [];
let gd_validation = [];
let color_in = null;
let first_time = true;
let last_performance = null;

function handle_message(data) {
	if (first_time) {
		color_in = setInterval(function() {
			const d = new Date();
			const x = (1 + Math.cos(d.getTime() / 1000)) * 30 + 40;
			document.getElementById("showcase_background").style.background = "rgb(0, 10, " + x + ")";
		}, 10);
		first_time = false;
		let nodatas = document.getElementsByClassName("no_data");
		for (let i = 0; i < nodatas.length; i++)
			nodatas[i].style.display = "none";
		document.getElementById("learning_data").style.display = "block";
		document.getElementById("model_information").style.display = "block";
		gd_epochs = [];
		gd_batches = [];
		gd_validation = [];

		document.getElementById("learning_pause").disabled = false;
		document.getElementById("learning_play").disabled = false;
		document.getElementById("learning_stop").disabled = false;

		let xmlHttp = new XMLHttpRequest();
		xmlHttp.onreadystatechange = function() {
			if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
				describe_model(xmlHttp.responseText);
			}
		}
		xmlHttp.open("GET", "http://localhost:5111/describe", true); // true for asynchronous 
		xmlHttp.send(null);
	}
	console.log(data);
	const msg = JSON.parse(data);
	for (let i = 0; i < msg.batches.length; i++)
		gd_batches.push({ x: i, y: msg.batches[i].error });
	for (let i = 0; i < msg.epochs.length; i++) {
		gd_epochs.push({ x: msg.epochs[i].epoch, y: msg.epochs[i].error });
		gd_validation.push({ x: msg.epochs[i].epoch, y: msg.epochs[i].validation_error });
	}
	if ('state' in msg) {
		if (msg.state === "play") {
			document.getElementById("learning_pause").style.display = "inline-block";
			document.getElementById("learning_play").style.display = "none";
		} else {
			document.getElementById("learning_pause").style.display = "none";
			document.getElementById("learning_play").style.display = "inline-block";
		}
	}
	if ('profiling' in msg) {
		if (msg.profiling) {
			document.getElementById("profiling_toggle").innerHTML = "stop profiling";
			document.getElementById("profiling_toggle").onclick = function() { toggle_profiling(false); };
		} else {
			document.getElementById("profiling_toggle").innerHTML = "start profiling";
			document.getElementById("profiling_toggle").onclick = function() { toggle_profiling(true); };
		}
	}
	if ('profiling_data' in msg) {
		last_performance = msg.profiling_data;
	}
	if (msg.batches.length > 0) {
		const last_batch = msg.batches[msg.batches.length - 1];
		document.getElementById("batch_counter").innerHTML = last_batch.batch + "/" + msg.total_batches;
	}
	if (msg.epochs.length > 0) {
		const last_epoch = msg.epochs[msg.epochs.length - 1];
		document.getElementById("epoch_counter").innerHTML = last_epoch.epoch;
	}
	visualize_data();
}

let chart = null;
let performance_pie = null;
function visualize_data() {
	let labels = gd_epochs.map((e) => e.x);
	if (chart === null) {
		const data = {
			labels: labels,
			datasets: [
				{
					label: 'Epoch Error',
					data: gd_epochs,
					borderColor: 'red'
				},
				{
					label: 'Validation Error',
					data: gd_validation,
					borderColor: 'blue'
				},
			]
		};
		const chart_options = {
			legend: {
				display: true
			}
		};
		chart = new Chart(document.getElementById("learning_curve"), {
			type: 'line',
			data: data,
			options: chart_options
		});
	} else {
		chart.data.labels = labels;
		chart.data.datasets[0].data = gd_epochs;
		chart.data.datasets[1].data = gd_validation;
		chart.update();
	}
	if (last_performance != null) {
		let layers = ["gradient"].concat(last_performance.forward.map((p, i) => (i + 1) + ": " + p.name));
		let times = [last_performance.gradient / 1000000.0].concat(last_performance.forward.map((p) => p.time / 1000000.0));
		if (performance_pie === null) {
			const data = {
				labels: layers,
				datasets: [
					{
						label: 'time per batch (ms)',
						data: times
					}]
			};
			performance_pie = new Chart(document.getElementById("performance_pie"), {
				type: 'pie',
				data: data,
				options: {
					plugins: {
						legend: {
							display: false
						},
					}
				}
			});
		} else {
			performance_pie.data.labels = layers;
			performance_pie.data.datasets[0].data = times;
			performance_pie.update();
		}
		let sum = times.reduce((pv, cv) => pv + cv, 0)
		document.getElementById("performance_total_time").innerHTML = "<b>Time/Batch:</b> " + Math.round(sum) + "ms"
	}
}

async function connect_and_receive() {
	let xmlHttp = new XMLHttpRequest();
	xmlHttp.onreadystatechange = function() {
		if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
			document.getElementById("connection_status").innerHTML = "Establish connection, receiving data";
			handle_message(xmlHttp.responseText);
		} else {
			if (xmlHttp.readyState == 4) {
				document.getElementById("connection_status").innerHTML = "Searching for a Connection ...";
				first_time = true;
				document.getElementById("learning_pause").disabled = true;
				document.getElementById("learning_play").disabled = true;
				document.getElementById("learning_stop").disabled = true;
				if (color_in != null)
					clearInterval(color_in);
			}
		}
	}
	xmlHttp.open("GET", "http://localhost:5111/" + id, true); // true for asynchronous 
	xmlHttp.send(null);
}
function pause_play_training(activate) {
	if (!activate) {
		document.getElementById("learning_pause").style.display = "none";
		document.getElementById("learning_play").style.display = "inline-block";
	} else {
		document.getElementById("learning_pause").style.display = "inline-block";
		document.getElementById("learning_play").style.display = "none";
	}
	let xmlHttp = new XMLHttpRequest();
	xmlHttp.open("GET", "http://localhost:5111/" + (activate ? "play" : "pause"), true); // true for asynchronous 
	xmlHttp.send(null);
}
function stop_training() {
	let xmlHttp = new XMLHttpRequest();
	xmlHttp.open("GET", "http://localhost:5111/stop", true); // true for asynchronous 
	xmlHttp.send(null);
	document.getElementById("learning_pause").disabled = true;
	document.getElementById("learning_play").disabled = true;
	document.getElementById("learning_stop").disabled = true;
	if (color_in != null)
		clearInterval(color_in);
}
function toggle_profiling(do_profiling) {
	let xmlHttp = new XMLHttpRequest();
	if (do_profiling) {
		xmlHttp.open("GET", "http://localhost:5111/start_profiling", true);
		document.getElementById("profiling_toggle").value = "stop profiling";
		document.getElementById("profiling_toggle").onclick = function() { toggle_profiling(false); };
	} else {
		xmlHttp.open("GET", "http://localhost:5111/stop_profiling", true);
		document.getElementById("profiling_toggle").value = "start profiling";
		document.getElementById("profiling_toggle").onclick = function() { toggle_profiling(true); };
	}
	xmlHttp.send(null);
}
function describe_model(data) {
	const description = JSON.parse(data);
	console.log(description);
	document.getElementById("model_summary").innerHTML = "";
	for (let i = 0; i < description.layers.length; i++) {
		const layer = description.layers[i];
		document.getElementById("model_summary").innerHTML +=
			'<div class="layer_card"><div class="layer_header"><b>' + layer.name +
			'</b>&nbsp;<pre style="display:inline;">' + layer.no_params +
			'</pre> parameters</div><div class="layer_description">' + layer.description + '</div></div>';
	}
	document.getElementById("optimizer_info").innerHTML = "";
	document.getElementById("optimizer_info").innerHTML += '<b>Loss Function: </b>' + description.loss_fct + "<br/>";
	document.getElementById("optimizer_info").innerHTML += '<div class="layer_card"><div class="layer_header"><b>'
		+ description.optimizer.name + '</b></div><div class="layer_description">' + description.optimizer.description + '</div></div>';
}
