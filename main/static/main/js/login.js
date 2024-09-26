document.addEventListener('DOMContentLoaded', function() {
  const signInBtn = document.getElementById("signIn");
  const signUpBtn = document.getElementById("signUp");
  const container = document.querySelector(".container");
  const signUpButton = document.querySelector(".container--signup .btn");
  const loginForm = document.getElementById("loginForm");

  signInBtn.addEventListener("click", () => {
    container.classList.remove("right-panel-active");
  });

  signUpBtn.addEventListener("click", () => {
    container.classList.add("right-panel-active");
  });

  signUpButton.addEventListener("click", showPopup);

  loginForm.addEventListener("submit", function(event) {
    event.preventDefault();
    const loginName = document.querySelector('input[name="login_name"]').value;
    const loginPhoneNumber = document.querySelector('input[name="login_phone_number"]').value;
    const signUpName = document.querySelector('input[name="name"]').value;
    const signUpPhoneNumber = document.querySelector('input[name="phone_number"]').value;

    if (loginName === signUpName && loginPhoneNumber === signUpPhoneNumber) {
      // The login details are correct, update the form action URL and submit the form
      loginForm.action = "index/";
      loginForm.submit();
    } else {
      // The login details are incorrect, display an error message or take appropriate action
      alert("로그인 정보가 올바르지 않습니다.");
    }
  });
});

function showPopup() {
  var popupContainer = document.getElementById('popupContainer');
  popupContainer.style.display = 'block';
}