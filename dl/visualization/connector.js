function sleep(milliseconds) {
	return new Promise(resolve => setTimeout(resolve, milliseconds));
}

const id = new Date().getTime();

let gd_epochs = [];
let gd_batches = [];
let gd_validation = [];
let color_in = null;
let first_time = true;

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
		gd_batches.push({x: i, y: msg.batches[i].error});
	for (let i = 0; i < msg.epochs.length; i++) {
		gd_epochs.push({x: msg.epochs[i].epoch, y: msg.epochs[i].error});
		gd_validation.push({x: msg.epochs[i].epoch, y: msg.epochs[i].validation_error});
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
function describe_model(data) {
  document.getElementById("model_information").innerHTML = "";
  const description = JSON.parse(data);
  console.log(description);
  for (let i = 0; i < description.layers.length; i++) {
    const layer = description.layers[i];
    document.getElementById("model_information").innerHTML += '<div class="layer_card"><div class="layer_header"><b>' + layer.name + 
      '</b>&nbsp;<pre style="display:inline;">' + layer.no_params + '</pre> parameters</div><div class="layer_description">' + layer.description + '</div></div>';
  }
}
