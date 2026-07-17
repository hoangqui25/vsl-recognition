// =====================================
// VSL Recognition - Frontend JavaScript
// =====================================

// ---------- Video Preview ----------

const videoInput = document.getElementById("videoInput");

const previewVideo = document.getElementById("previewVideo");

const previewContainer = document.getElementById("previewContainer");

if (videoInput) {
  videoInput.addEventListener("change", function () {
    const file = this.files[0];

    if (file) {
      // Kiểm tra định dạng video
      if (!file.type.startsWith("video/")) {
        alert("Please upload a video file!");

        this.value = "";

        return;
      }

      // Tạo URL preview
      const videoURL = URL.createObjectURL(file);

      previewVideo.src = videoURL;

      previewContainer.style.display = "block";
    }
  });
}

// ---------- Submit Loading Effect ----------

const form = document.querySelector("form");

if (form) {
  form.addEventListener("submit", function () {
    const button = document.getElementById("predictBtn");

    if (button) {
      button.disabled = true;

      button.innerHTML = `

                <span 
                class="spinner-border spinner-border-sm me-2">
                </span>

                Processing...

            `;
    }
  });
}

// ---------- Drag & Drop Upload ----------

const uploadBox = document.querySelector(".form-control");

if (uploadBox) {
  uploadBox.addEventListener("dragover", function (event) {
    event.preventDefault();

    this.classList.add("border-primary");
  });

  uploadBox.addEventListener("dragleave", function () {
    this.classList.remove("border-primary");
  });
}

// ---------- Confidence Animation ----------

const progressBar = document.querySelector(".progress-bar");

if (progressBar) {
  const confidence = progressBar.style.width;

  progressBar.style.width = "0%";

  setTimeout(() => {
    progressBar.style.width = confidence;
  }, 300);
}

// ---------- Auto hide alerts ----------

const alerts = document.querySelectorAll(".alert");

alerts.forEach((alert) => {
  setTimeout(() => {
    alert.style.opacity = "0";

    setTimeout(() => {
      alert.remove();
    }, 500);
  }, 5000);
});
