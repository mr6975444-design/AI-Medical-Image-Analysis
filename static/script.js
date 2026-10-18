const imageInput = document.getElementById("imageInput");
const chooseButton = document.getElementById("chooseButton");
const uploadArea = document.getElementById("uploadArea");
const analyzeButton = document.getElementById("analyzeButton");

const previewContainer =
    document.getElementById("previewContainer");

const imagePreview =
    document.getElementById("imagePreview");

const resultCard =
    document.getElementById("resultCard");

const prediction =
    document.getElementById("prediction");

const confidence =
    document.getElementById("confidence");

const confidenceFill =
    document.getElementById("confidenceFill");

const resultDisclaimer =
    document.getElementById("resultDisclaimer");

let selectedFile = null;


chooseButton.addEventListener("click", function () {
    imageInput.click();
});


uploadArea.addEventListener("click", function (event) {

    if (event.target === chooseButton) {
        return;
    }

    imageInput.click();

});


imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (file) {
        handleFile(file);
    }

});


uploadArea.addEventListener("dragover", function (event) {

    event.preventDefault();

    uploadArea.classList.add("dragover");

});


uploadArea.addEventListener("dragleave", function () {

    uploadArea.classList.remove("dragover");

});


uploadArea.addEventListener("drop", function (event) {

    event.preventDefault();

    uploadArea.classList.remove("dragover");

    const file = event.dataTransfer.files[0];

    if (file) {
        handleFile(file);
    }

});


function handleFile(file) {

    if (!file.type.startsWith("image/")) {

        alert("Please select an image file.");

        return;
    }

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = function (event) {

        imagePreview.src = event.target.result;

        previewContainer.classList.remove("hidden");

        analyzeButton.disabled = false;

        resultCard.classList.add("hidden");

    };

    reader.readAsDataURL(file);
}


analyzeButton.addEventListener("click", async function () {

    if (!selectedFile) {
        return;
    }

    analyzeButton.disabled = true;

    analyzeButton.textContent = "Analyzing...";

    const formData = new FormData();

    formData.append("image", selectedFile);

    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Prediction failed."
            );
        }

        showResult(data);

    } catch (error) {

        alert(error.message);

    } finally {

        analyzeButton.disabled = false;

        analyzeButton.textContent = "Analyze Image";

    }

});


function showResult(data) {

    resultCard.classList.remove("hidden");

    prediction.textContent = data.prediction;

    const percentage =
        data.model_confidence * 100;

    confidence.textContent =
        percentage.toFixed(2) + "%";

    confidenceFill.style.width =
        percentage + "%";

    resultDisclaimer.textContent =
        data.disclaimer;

    resultCard.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });

}