// NeuralAI landing page — animated neural mesh + scroll reveals + stat counters
(function () {
  // ---- Animated neural mesh canvas ----
  const canvas = document.getElementById('neural-canvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    let w, h, nodes = [];
    const COLORS = ['#1a73e8', '#ea4335', '#fbbc04', '#34a853'];
    function resize() {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
    }
    function init() {
      const count = Math.min(70, Math.floor((w * h) / 26000));
      nodes = [];
      for (let i = 0; i < count; i++) {
        nodes.push({
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * 0.35,
          vy: (Math.random() - 0.5) * 0.35,
          c: COLORS[i % COLORS.length],
          r: Math.random() * 1.8 + 1
        });
      }
    }
    function draw() {
      ctx.clearRect(0, 0, w, h);
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > w) n.vx *= -1;
        if (n.y < 0 || n.y > h) n.vy *= -1;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = n.c;
        ctx.globalAlpha = 0.8;
        ctx.fill();
        for (let j = i + 1; j < nodes.length; j++) {
          const m = nodes[j];
          const dx = n.x - m.x, dy = n.y - m.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 130) {
            ctx.beginPath();
            ctx.moveTo(n.x, n.y);
            ctx.lineTo(m.x, m.y);
            ctx.strokeStyle = n.c;
            ctx.globalAlpha = (1 - dist / 130) * 0.18;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }
      ctx.globalAlpha = 1;
      requestAnimationFrame(draw);
    }
    resize();
    init();
    draw();
    window.addEventListener('resize', function () { resize(); init(); });
  }

  // ---- Scroll reveal ----
  const reveals = document.querySelectorAll('.section, .hero-inner, .cta-section');
  reveals.forEach(function (el) { el.classList.add('reveal'); });
  const io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    });
  }, { threshold: 0.12 });
  reveals.forEach(function (el) { io.observe(el); });

  // ---- Stat counters ----
  const stats = document.querySelectorAll('[data-count]');
  const sio = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      const el = e.target;
      const target = parseInt(el.getAttribute('data-count'), 10);
      let cur = 0;
      const step = Math.max(1, Math.floor(target / 40));
      const tick = setInterval(function () {
        cur += step;
        if (cur >= target) { cur = target; clearInterval(tick); }
        el.textContent = cur + (target >= 50 ? '%' : '');
      }, 28);
      sio.unobserve(el);
    });
  }, { threshold: 0.5 });
  stats.forEach(function (el) { sio.observe(el); });
})();
