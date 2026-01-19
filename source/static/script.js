document.addEventListener("DOMContentLoaded", function() {
    console.log("BrainScan AI Loaded 🚀");
    
    // Add a simple animation effect to buttons
    document.querySelectorAll('.btn').forEach(button => {
        button.addEventListener('mouseover', function() {
            this.style.transform = "scale(1.1)";
        });
        button.addEventListener('mouseout', function() {
            this.style.transform = "scale(1)";
        });
    });

});
document.addEventListener("DOMContentLoaded", function () {
    // 🟢 Smooth Scroll Function
    document.querySelectorAll("a[href^='#']").forEach(anchor => {
        anchor.addEventListener("click", function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute("href"));
            if (target) {
                target.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }
        });
    });
});
document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("#upload-form");
    const submitButton = document.querySelector("#predict-btn");
    const loader = document.querySelector("#loading-animation");

    form.addEventListener("submit", function () {
        submitButton.disabled = true; // 🔒 Disable button to prevent multiple clicks
        loader.style.display = "block"; // 🔄 Show the loading animation
    });
});
