document.addEventListener(
    "DOMContentLoaded",
    function () {

        /*
        ========================================================
        FILE UPLOAD / PREVIEW
        ========================================================
        */

        const dropzone =
            document.getElementById("dropzone");

        const fileInput =
            document.getElementById("fileInput");

        const previewWrap =
            document.getElementById("previewWrap");

        const previewThumb =
            document.getElementById("previewThumb");

        const fileNameEl =
            document.getElementById("fileName");

        const fileSizeEl =
            document.getElementById("fileSize");

        const uploadForm =
            document.getElementById("uploadForm");

        const analyzeBtn =
            document.getElementById("analyzeBtn");

        const analyzeBtnText =
            document.getElementById("analyzeBtnText");


        function formatBytes(bytes) {

            if (bytes < 1024) {
                return bytes + " B";
            }

            if (bytes < 1024 * 1024) {

                return (
                    bytes / 1024
                ).toFixed(1) + " KB";

            }

            return (
                bytes /
                (1024 * 1024)
            ).toFixed(2) + " MB";

        }


        function showPreview(file) {

            if (!file) {
                return;
            }

            if (!file.type.startsWith("image/")) {

                alert(
                    "Please select a valid image file."
                );

                return;
            }

            const reader =
                new FileReader();

            reader.onload =
                function (event) {

                    if (previewThumb) {

                        previewThumb.src =
                            event.target.result;

                    }

                    if (fileNameEl) {

                        fileNameEl.textContent =
                            file.name;

                    }

                    if (fileSizeEl) {

                        fileSizeEl.textContent =
                            formatBytes(
                                file.size
                            );

                    }

                    if (previewWrap) {

                        previewWrap.classList.add(
                            "active"
                        );

                    }

                };

            reader.readAsDataURL(file);

        }


        if (fileInput) {

            fileInput.addEventListener(
                "change",
                function () {

                    if (
                        fileInput.files.length > 0
                    ) {

                        showPreview(
                            fileInput.files[0]
                        );

                    }

                }
            );

        }


        /*
        ========================================================
        DRAG AND DROP
        ========================================================
        */

        if (dropzone) {

            [
                "dragenter",
                "dragover"
            ].forEach(
                function (eventName) {

                    dropzone.addEventListener(
                        eventName,
                        function (event) {

                            event.preventDefault();

                            event.stopPropagation();

                            dropzone.classList.add(
                                "dragover"
                            );

                        }
                    );

                }
            );


            [
                "dragleave",
                "drop"
            ].forEach(
                function (eventName) {

                    dropzone.addEventListener(
                        eventName,
                        function (event) {

                            event.preventDefault();

                            event.stopPropagation();

                            dropzone.classList.remove(
                                "dragover"
                            );

                        }
                    );

                }
            );


            dropzone.addEventListener(
                "drop",
                function (event) {

                    const files =
                        event.dataTransfer.files;

                    if (
                        files.length > 0
                    ) {

                        try {

                            const dataTransfer =
                                new DataTransfer();

                            dataTransfer.items.add(
                                files[0]
                            );

                            fileInput.files =
                                dataTransfer.files;

                        } catch (error) {

                            console.log(
                                "DataTransfer unavailable."
                            );

                        }

                        showPreview(
                            files[0]
                        );

                    }

                }
            );

        }


        /*
        ========================================================
        FORM SUBMISSION
        ========================================================
        */

        if (uploadForm) {

            uploadForm.addEventListener(
                "submit",
                function () {

                    if (
                        !fileInput ||
                        fileInput.files.length === 0
                    ) {

                        return;

                    }

                    if (analyzeBtn) {

                        analyzeBtn.classList.add(
                            "loading"
                        );

                        analyzeBtn.disabled = true;

                    }

                    if (analyzeBtnText) {

                        analyzeBtnText.textContent =
                            "Analyzing...";

                    }

                }
            );

        }


        /*
        ========================================================
        CONFIDENCE RING
        ========================================================
        */

        const ring =
            document.getElementById(
                "ringFill"
            );

        if (ring) {

            const radius = 46;

            const circumference =
                2 *
                Math.PI *
                radius;

            let confidence =
                parseFloat(
                    ring.dataset.confidence
                );

            if (isNaN(confidence)) {

                confidence = 0;

            }

            confidence =
                Math.max(
                    0,
                    Math.min(
                        100,
                        confidence
                    )
                );

            ring.style.strokeDasharray =
                circumference;

            ring.style.strokeDashoffset =
                circumference;

            requestAnimationFrame(
                function () {

                    const offset =
                        circumference -
                        (
                            confidence /
                            100
                        ) *
                        circumference;

                    ring.style.strokeDashoffset =
                        offset;

                }
            );

        }


        /*
        ========================================================
        PROBABILITY BARS
        ========================================================
        */

        const probabilityFills =
            document.querySelectorAll(
                ".probability-fill"
            );

        probabilityFills.forEach(
            function (bar) {

                let value =
                    parseFloat(
                        bar.dataset.width
                    );

                if (isNaN(value)) {

                    value = 0;

                }

                value =
                    Math.max(
                        0,
                        Math.min(
                            100,
                            value
                        )
                    );

                bar.style.width =
                    "0%";

                requestAnimationFrame(
                    function () {

                        bar.style.width =
                            value + "%";

                    }
                );

            }
        );

    }
);