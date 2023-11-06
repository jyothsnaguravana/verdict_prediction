let verdictRequest = document.getElementById('verdictRequest');

verdictRequest.addEventListener('click', function(e) {
  e.preventDefault();
  verdictRequest.disabled = true;
  verdictRequest.value = 'Wait...';
  let verdictElement = document.getElementById('verdictResponse');
  verdictElement.innerHTML = "Reading Inputs...";
  let outputImgElement = document.getElementById('output-img');
  let petitioner_name = document.getElementById('petitionerName').value;
  let respondent_name = document.getElementById('respondentName').value;
  let facts = document.getElementById('factsCollected').value;
  outputImgElement.src = "static/assets/thinking.gif";
  verdictElement.innerHTML = "Calculating Verdict...";

  fetch('/verdict', {
    method: 'POST',
    body: JSON.stringify({
      petitioner_name: petitioner_name,
      respondent_name: respondent_name,
      facts: facts
    }),
    headers: {
      'Content-Type': 'application/json'
    }
  }).then(function(response) {
    if(response.ok) {
      response.json().then(function(data) {
        outputImgElement.src = data['image'];
        let verdict = data['verdict'];
        verdictElement.innerHTML = verdict;
        verdictRequest.disabled = false;
        verdictRequest.value = 'Get Verdict...';
      });
    } else {
      throw new Error('Something went wrong');
    }
  })
  .catch(function(error) {
    console.log(error);
    verdictRequest.value = 'Error...Refresh Page';
  });
});