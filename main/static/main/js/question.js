let slideIndex = 1;
showSlides(slideIndex);

function plusSlides(n) {
  showSlides(slideIndex += n);
}

function showSlides(n) {
  let slides = document.getElementsByClassName("slide");
  let prevButton = document.querySelector(".prev");
  let nextButton = document.querySelector(".next");

  // 슬라이드 인덱스 범위 제한
  if (n < 1) {
    slideIndex = slides.length;
  } else if (n > slides.length) {
    slideIndex = 1;
  } else {
    slideIndex = n;
  }

  // 모든 슬라이드 숨기기
  for (let i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";
  }

  // 현재 슬라이드 보이기
  slides[slideIndex - 1].style.display = "block";

  // 이전/다음 버튼 보이기/숨기기
  if (slideIndex === 1) {
    prevButton.style.display = "none";
  } else {
    prevButton.style.display = "block";
  }

  if (slideIndex === slides.length) {
    nextButton.style.display = "none";
  } else {
    nextButton.style.display = "block";
  }
}

