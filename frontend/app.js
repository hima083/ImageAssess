const API = "http://127.0.0.1:8000";

const imageInput = document.getElementById("imageInput");
const previewBox = document.getElementById("previewBox");
const analyzeBtn = document.getElementById("analyzeBtn");

let selectedFile = null;


// ============================================================
// IMAGE SELECTION
// ============================================================

previewBox.onclick = () => {
    imageInput.click();
};


imageInput.onchange = () => {

    selectedFile = imageInput.files[0];

    if (!selectedFile) {
        return;
    }

    const reader = new FileReader();

    reader.onload = event => {

        previewBox.innerHTML = `
            <img
                src="${event.target.result}"
                alt="Selected image"
                style="
                    max-width:100%;
                    max-height:300px;
                    border-radius:12px;
                    object-fit:contain;
                "
            >

            <p>${selectedFile.name}</p>
        `;
    };

    reader.readAsDataURL(selectedFile);
};


// ============================================================
// ANALYZE IMAGE
// ============================================================

analyzeBtn.onclick = async () => {

    if (!selectedFile) {

        showError("Please select an image first.");

        return;
    }


    const loading =
        document.getElementById("loading");

    const result =
        document.getElementById("result");

    const error =
        document.getElementById("error");


    // --------------------------------------------------------
    // Reset UI
    // --------------------------------------------------------

    loading.classList.remove("hidden");

    error.textContent = "";

    analyzeBtn.disabled = true;

    analyzeBtn.textContent = "Analyzing...";


    // IMPORTANT:
    // Show the result section immediately.
    // This prevents the UI from looking like nothing happened.

    result.classList.remove("hidden");


    document.getElementById(
        "qualityLabel"
    ).textContent = "ANALYZING...";


    document.getElementById(
        "qualityScore"
    ).textContent = "—";


    document.getElementById(
        "issues"
    ).innerHTML = `
        <div class="issue">
            <strong>Analyzing image</strong>
            <span>
                The ML engine is checking image quality...
            </span>
        </div>
    `;


    document.getElementById(
        "confidence"
    ).innerHTML = `
        <div class="stat">
            <small>Status</small>
            <strong>Processing...</strong>
        </div>
    `;


    document.getElementById(
        "statistics"
    ).innerHTML = `
        <div class="stat">
            <small>Status</small>
            <strong>Processing...</strong>
        </div>
    `;


    // --------------------------------------------------------
    // Prepare image
    // --------------------------------------------------------

    const formData = new FormData();

    formData.append(
        "file",
        selectedFile
    );


    try {

        // ----------------------------------------------------
        // Send image to FastAPI
        // ----------------------------------------------------

        const response = await fetch(
            `${API}/analyze`,
            {
                method: "POST",
                body: formData
            }
        );


        // ----------------------------------------------------
        // Read backend response
        // ----------------------------------------------------

        const data =
            await response.json();


        console.log(
            "ANALYSIS RESPONSE:",
            data
        );


        // ----------------------------------------------------
        // Check response
        // ----------------------------------------------------

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Analysis failed."
            );
        }


        // ----------------------------------------------------
        // Display ML result
        // ----------------------------------------------------

        displayResult(data);


        // ----------------------------------------------------
        // Refresh history
        // ----------------------------------------------------

        await loadHistory();


    } catch (error) {

        console.error(
            "Analysis error:",
            error
        );


        result.classList.add("hidden");


        showError(
            error.message ||
            "Could not analyze image."
        );


    } finally {

        loading.classList.add("hidden");

        analyzeBtn.disabled = false;

        analyzeBtn.textContent =
            "Analyze Image";
    }
};


// ============================================================
// DISPLAY RESULT
// ============================================================

function displayResult(data) {

    const result =
        document.getElementById("result");


    // --------------------------------------------------------
    // Make result visible
    // --------------------------------------------------------

    result.classList.remove("hidden");


    // ========================================================
    // QUALITY LABEL
    // ========================================================

    document.getElementById(
        "qualityLabel"
    ).textContent =
        formatLabel(
            data.quality_label ||
            "UNKNOWN"
        );


    // ========================================================
    // QUALITY SCORE
    // ========================================================

    document.getElementById(
        "qualityScore"
    ).textContent =
        Number(
            data.quality_score || 0
        ).toFixed(1);


    // ========================================================
    // DETECTED ISSUES
    // ========================================================

    const issuesBox =
        document.getElementById(
            "issues"
        );


    if (
        !data.issues ||
        data.issues.length === 0
    ) {

        issuesBox.innerHTML = `
            <div class="issue">

                <strong>
                    No major issues
                </strong>

                <span>
                    The image passed the diagnostic checks.
                </span>

            </div>
        `;

    } else {

        issuesBox.innerHTML =
            data.issues
            .map(issue => {

                const confidence =
                    Number(
                        issue.confidence || 0
                    ) * 100;


                return `
                    <div class="issue">

                        <strong>
                            ${formatLabel(
                                issue.type
                            )}
                        </strong>

                        <span>

                            Severity:
                            ${formatLabel(
                                issue.severity
                            )}

                            ·

                            Confidence:
                            ${confidence.toFixed(0)}%

                        </span>

                    </div>
                `;

            })
            .join("");
    }


    // ========================================================
    // PREDICTION CONFIDENCE
    // ========================================================

    const confidenceBox =
        document.getElementById(
            "confidence"
        );


    const probabilities =
        data.probabilities || {};


    confidenceBox.innerHTML =
        Object.entries(probabilities)
        .map(([label, value]) => {

            return `
                <div class="stat">

                    <small>
                        ${formatLabel(label)}
                    </small>

                    <strong>
                        ${(Number(value) * 100).toFixed(1)}%
                    </strong>

                </div>
            `;

        })
        .join("");


    // ========================================================
    // IMAGE STATISTICS
    // ========================================================

    const statisticsBox =
        document.getElementById(
            "statistics"
        );


    const statistics =
        data.features || {};


    if (
        Object.keys(statistics).length === 0
    ) {

        statisticsBox.innerHTML = `
            <div class="stat">

                <small>
                    Feature data
                </small>

                <strong>
                    Not available
                </strong>

            </div>
        `;

    } else {

        statisticsBox.innerHTML =
            Object.entries(statistics)
            .map(([name, value]) => {

                let displayValue;


                if (
                    typeof value === "number"
                ) {

                    displayValue =
                        value.toFixed(3);

                } else {

                    displayValue =
                        String(value);
                }


                return `
                    <div class="stat">

                        <small>
                            ${formatLabel(name)}
                        </small>

                        <strong>
                            ${displayValue}
                        </strong>

                    </div>
                `;

            })
            .join("");
    }


    // ========================================================
    // SCROLL TO RESULT
    // ========================================================

    setTimeout(() => {

        result.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }, 100);
}


// ============================================================
// FORMAT LABEL
// ============================================================

function formatLabel(value) {

    return String(value)
        .replaceAll("_", " ")
        .toLowerCase()
        .replace(
            /\b\w/g,
            char => char.toUpperCase()
        );
}


// ============================================================
// LOAD HISTORY
// ============================================================

async function loadHistory() {

    try {

        const response =
            await fetch(
                `${API}/analyses`
            );


        if (!response.ok) {

            throw new Error(
                "Could not load history."
            );
        }


        const data =
            await response.json();


        const history =
            document.getElementById(
                "history"
            );


        if (
            !data ||
            data.length === 0
        ) {

            history.innerHTML =
                "<p>No analyses yet.</p>";

            return;
        }


        history.innerHTML =
            data.map(item => {

                return `
                    <div class="history-item">

                        <span>
                            ${item.filename}
                        </span>

                        <strong>
                            ${formatLabel(
                                item.quality_label
                            )}

                            ·

                            ${Number(
                                item.quality_score || 0
                            ).toFixed(1)}

                        </strong>

                    </div>
                `;

            })
            .join("");


    } catch (error) {

        console.error(
            "Could not load history:",
            error
        );
    }
}


// ============================================================
// ERROR MESSAGE
// ============================================================

function showError(message) {

    document.getElementById(
        "error"
    ).textContent = message;
}


// ============================================================
// HISTORY BUTTON
// ============================================================

document.getElementById(
    "historyBtn"
).onclick = loadHistory;


// ============================================================
// INITIAL LOAD
// ============================================================

loadHistory();