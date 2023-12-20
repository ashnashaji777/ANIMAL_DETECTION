function detect() {
  // Your detection logic here

  // For demo purposes, let's assume an animal is detected
  let animalDetected = true;

  if (animalDetected) {
      displayAlert('Animal detected! Please be cautious.');
      playAlertSound();
  }
}

function displayAlert(message) {
  let alertBox = document.getElementById('alertBox');
  let alertMessage = document.getElementById('alertMessage');

  alertMessage.innerText = message;
  alertBox.style.display = 'block';
}

function playAlertSound() {
  let alertSound = document.getElementById('alertSound');
  alertSound.play();
}
