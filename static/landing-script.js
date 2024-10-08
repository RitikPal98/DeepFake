// Smooth scrolling for nav links with easing effect
document.querySelectorAll('nav ul li a').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));

        // Smooth scrolling with easing
        window.scrollTo({
            top: target.offsetTop,
            behavior: 'smooth'
        });
    });
});

// Add scroll-triggered animations for sections
const sections = document.querySelectorAll('section');
const options = {
    root: null,
    threshold: 0.1,
    rootMargin: '0px'
};

const observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (!entry.isIntersecting) return;

        entry.target.classList.add('fade-in');
        observer.unobserve(entry.target);
    });
}, options);

sections.forEach(section => {
    observer.observe(section);
});

// Smooth animations for different elements
const fadeInElements = document.querySelectorAll('.fade-in-on-scroll');

fadeInElements.forEach(el => {
    observer.observe(el);
});

// Button hover animations
const buttons = document.querySelectorAll('.cta-btn, .hero-btn, .contact-form button');

buttons.forEach(button => {
    button.addEventListener('mouseenter', () => {
        button.classList.add('button-hover');
    });

    button.addEventListener('mouseleave', () => {
        button.classList.remove('button-hover');
    });
});

// Interactive contact form feedback
const form = document.querySelector('.contact-form');
form.addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Simulate form submission and show success message
    const successMessage = document.createElement('p');
    successMessage.textContent = "Your message has been successfully sent!";
    successMessage.style.color = "#ff4b2b";
    successMessage.style.textAlign = "center";
    
    form.parentElement.appendChild(successMessage);

    // Fade out success message after 3 seconds
    setTimeout(() => {
        successMessage.style.opacity = 0;
        setTimeout(() => successMessage.remove(), 1000);
    }, 3000);

    // Reset form
    form.reset();
});

// Add staggered animations to the team members on load
const teamMembers = document.querySelectorAll('.team-member');
teamMembers.forEach((member, index) => {
    setTimeout(() => {
        member.classList.add('fade-in');
    }, index * 300); // Delay each member’s appearance
});

// Scroll back to top button
const scrollToTopBtn = document.createElement('button');
scrollToTopBtn.innerText = '⬆ Back to Top';
scrollToTopBtn.classList.add('scroll-to-top');
document.body.appendChild(scrollToTopBtn);

window.addEventListener('scroll', () => {
    if (window.scrollY > 500) {
        scrollToTopBtn.style.opacity = '1';
    } else {
        scrollToTopBtn.style.opacity = '0';
    }
});

scrollToTopBtn.addEventListener('click', () => {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
});

document.querySelectorAll('.faq-question').forEach(item => {
    item.addEventListener('click', () => {
        const parent = item.parentElement;

        // Toggle 'active' class to open/close FAQ answer
        parent.classList.toggle('active');
    });
});
// Smooth scroll when clicking on navbar items
document.querySelectorAll('nav a').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const targetId = this.getAttribute('href').substring(1);
        const targetSection = document.getElementById(targetId);
        window.scrollTo({
            top: targetSection.offsetTop,
            behavior: 'smooth'
        });
    });
});
// Hamburger Menu Functionality
const hamburger = document.getElementById("hamburger");
const navLinks = document.querySelector(".nav-links");

hamburger.addEventListener("click", () => {
    navLinks.classList.toggle("active");
});

// FAQ Toggle Functionality
const faqItems = document.querySelectorAll('.faq-item');

faqItems.forEach(item => {
    const question = item.querySelector('.faq-question');
    question.addEventListener('click', () => {
        item.classList.toggle('active');
    });
});

// Contact Form Submission Confirmation
document.querySelector('.contact-form').addEventListener('submit', function (event) {
    event.preventDefault();
    const formResponse = document.getElementById('form-response');
    formResponse.textContent = "Thank you for your message! We will get back to you shortly.";
    this.reset();
});


// Optional: Add interactive hover effect for technologies on mouseover
const techItems = document.querySelectorAll('.tech-item');

techItems.forEach(item => {
    item.addEventListener('mouseover', () => {
        item.style.transform = 'scale(1.2)';
    });
    item.addEventListener('mouseout', () => {
        item.style.transform = 'scale(1)';
    });
});


